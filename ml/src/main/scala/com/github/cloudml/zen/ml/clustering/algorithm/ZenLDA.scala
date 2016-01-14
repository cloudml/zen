/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.github.cloudml.zen.ml.clustering.algorithm

import java.util.Random
import java.util.concurrent.{ConcurrentLinkedQueue, Executors}

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV}
import com.github.cloudml.zen.ml.clustering.LDADefines._
import com.github.cloudml.zen.ml.sampler._
import com.github.cloudml.zen.ml.util.XORShiftRandom
import org.apache.spark.graphx2.impl.EdgePartition

import scala.collection.JavaConversions._
import scala.concurrent._
import scala.concurrent.duration._


class ZenLDA(numTopics: Int, numThreads: Int)
  extends LDATrainerByWord(numTopics, numThreads) {
  override def samplePartition(accelMethod: String,
    numPartitions: Int,
    sampIter: Int,
    seed: Int,
    topicCounters: BDV[Count],
    numTokens: Long,
    numTerms: Int,
    alpha: Double,
    alphaAS: Double,
    beta: Double)
    (pid: Int, ep: EdgePartition[TA, Nvk]): EdgePartition[TA, Int] = {
    val alphaRatio = alpha * numTopics / (numTokens + alphaAS * numTopics)
    val betaSum = beta * numTerms
    val denoms = calc_denoms(topicCounters, betaSum)
    val alphak_denoms = calc_alphak_denoms(denoms, alphaAS, betaSum, alphaRatio)
    val beta_denoms = denoms.copy :*= beta
    val totalSize = ep.size
    val lcSrcIds = ep.localSrcIds
    val lcDstIds = ep.localDstIds
    val vattrs = ep.vertexAttrs
    val data = ep.data
    val activeLens = new Array[Int](vattrs.length)
    val thq = new ConcurrentLinkedQueue(0 until numThreads)
    // table/ftree is a per term data structure
    // in GraphX, edges in a partition are clustered by source IDs (term id in this case)
    // so, use below simple cache to avoid calculating table each time
    val global: DiscreteSampler[Double] = accelMethod match {
      case "ftree" => new FTree[Double](isSparse=false)
      case "alias" | "hybrid" => new AliasTable
    }
    val gens = new Array[XORShiftRandom](numThreads)
    val termDists = new Array[DiscreteSampler[Double]](numThreads)
    val cdfDists = new Array[CumulativeDist[Double]](numThreads)
    resetDist_abDense(global, alphak_denoms, beta)

    implicit val es = ExecutionContext.fromExecutorService(Executors.newFixedThreadPool(numThreads))
    val all = Future.traverse(ep.index.iterator.zipWithIndex) { case ((_, startPos), ii) => Future {
      val thid = thq.poll()
      var gen = gens(thid)
      if (gen == null) {
        gen = new XORShiftRandom(((seed + sampIter) * numPartitions + pid) * numThreads + thid)
        gens(thid) = gen
        termDists(thid) = accelMethod match {
          case "alias" => new AliasTable[Double]
          case "ftree" | "hybrid" => new FTree(isSparse=true)
        }
        cdfDists(thid) = new CumulativeDist[Double]
        termDists(thid).reset(numTopics)
        cdfDists(thid).reset(numTopics)
      }
      val termDist = termDists(thid)
      val cdfDist = cdfDists(thid)

      val lsi = ii << 1
      val si = lcSrcIds(lsi)
      val endPos = lcSrcIds(lsi + 1)
      val termTopics = vattrs(si)
      activeLens(si) = termTopics.activeSize
      resetDist_waSparse(termDist, alphak_denoms, termTopics)
      val denseTermTopics = termTopics match {
        case v: BDV[Count] => v
        case v: BSV[Count] => toBDV(v)
      }
      val termBeta_denoms = calc_termBeta_denoms(denoms, beta_denoms, termTopics)
      var pos = startPos
      while (pos < endPos) {
        val di = lcDstIds(pos)
        val docTopics = vattrs(di).asInstanceOf[BSV[Count]]
        val topic = data(pos)
        resetDist_dwbSparse_withAdjust(cdfDist, denoms, termBeta_denoms, docTopics, topic)
        data(pos) = tokenSampling(gen, global, termDist, cdfDist, denseTermTopics, topic)
        pos += 1
      }
      thq.add(thid)
    }}
    Await.ready(all, 2.hour)
    es.shutdown()
    ep.withVertexAttributes(activeLens)
  }

  def tokenSampling(gen: Random,
    ab: DiscreteSampler[Double],
    wa: DiscreteSampler[Double],
    dwb: CumulativeDist[Double],
    termTopics: BDV[Count],
    topic: Int): Int = {
    val dwbSum = dwb.norm
    val sum23 = dwbSum + wa.norm
    val distSum = sum23 + ab.norm
    val genSum = gen.nextDouble() * distSum
    if (genSum < dwbSum) {
      dwb.sampleFrom(genSum, gen)
    } else if (genSum < sum23) wa match {
      case wt: AliasTable[Double] =>
        val rr = 1.0 / termTopics(topic)
        wt.resampleFrom(genSum - dwbSum, gen, topic, rr)
      case wf: FTree[Double] => wf.sampleFrom(genSum - dwbSum, gen)
    } else {
      ab.sampleFrom(genSum - sum23, gen)
    }
  }

  def tokenResampling(gen: Random,
    ab: DiscreteSampler[Double],
    wa: DiscreteSampler[Double],
    dwb: CumulativeDist[Double],
    termTopics: BDV[Count],
    docTopics: Ndk,
    topic: Int,
    beta: Double): Int = {
    val dwbSum = dwb.norm
    val sum23 = dwbSum + wa.norm
    val distSum = sum23 + ab.norm
    val genSum = gen.nextDouble() * distSum
    if (genSum < dwbSum) {
      val nkd = docTopics(topic)
      val nkw_beta = termTopics(topic) + beta
      val rr = 1.0 / nkd + 1.0 / nkw_beta - 1.0 / nkd / nkw_beta
      dwb.resampleFrom(genSum, gen, topic, rr)
    } else if (genSum < sum23) wa match {
      case wt: AliasTable[Double] =>
        val rr = 1.0 / termTopics(topic)
        wt.resampleFrom(genSum - dwbSum, gen, topic, rr)
      case wf: FTree[Double] => wf.sampleFrom(genSum - dwbSum, gen)
    } else {
      ab.sampleFrom(genSum - sum23, gen)
    }
  }
}
