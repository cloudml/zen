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
import java.util.concurrent.ConcurrentLinkedQueue

import breeze.linalg.{DenseVector => BDV}
import com.github.cloudml.zen.ml.clustering.LDADefines._
import com.github.cloudml.zen.ml.sampler._
import com.github.cloudml.zen.ml.util.Concurrent._
import com.github.cloudml.zen.ml.util.XORShiftRandom
import org.apache.spark.graphx2.impl.EdgePartition

import scala.collection.JavaConversions._
import scala.concurrent.Future


class FPlusLDA(numTopics: Int, numThreads: Int)
  extends LDATrainerByWord(numTopics, numThreads) {
  override def samplePartition(numPartitions: Int,
    sampIter: Int,
    seed: Int,
    topicCounters: BDV[Count],
    numTokens: Long,
    numTerms: Int,
    alpha: Double,
    alphaAS: Double,
    beta: Double)
    (pid: Int, ep: EdgePartition[TA, Nvk]): EdgePartition[TA, Int] = {
    val alphaSum = alpha * numTopics
    val betaSum = beta * numTerms
    val alphaRatio = calc_alphaRatio(alphaSum, numTokens, alphaAS)
    val denoms = calc_denoms(topicCounters, betaSum)
    val alphak_denoms = calc_alphak_denoms(denoms, alphaAS, betaSum, alphaRatio)

    val lcSrcIds = ep.localSrcIds
    val lcDstIds = ep.localDstIds
    val vattrs = ep.vertexAttrs
    val data = ep.data
    val useds = new Array[Int](vattrs.length)

    val global = new FTree[Double](isSparse=false)
    val thq = new ConcurrentLinkedQueue(1 to numThreads)
    val gens = new Array[XORShiftRandom](numThreads)
    val termDists = new Array[DiscreteSampler[Double]](numThreads)
    val cdfDists = new Array[CumulativeDist[Double]](numThreads)
    resetDist_abDense(global, topicCounters, alphaAS, beta, betaSum, alphaRatio)

    implicit val es = initExecutionContext(numThreads)
    val all = Future.traverse(lcSrcIds.indices.by(3).iterator)(lsi => withFuture {
      val thid = thq.poll() - 1
      try {
        var gen = gens(thid)
        if (gen == null) {
          gen = new XORShiftRandom(((seed + sampIter) * numPartitions + pid) * numThreads + thid)
          gens(thid) = gen
          termDists(thid) = new FTree[Double](isSparse = true).reset(numTopics)
          cdfDists(thid) = new CumulativeDist[Double].reset(numTopics)
        }
        val termDist = termDists(thid)
        val cdfDist = cdfDists(thid)

        val si = lcSrcIds(lsi)
        val startPos = lcSrcIds(lsi + 1)
        val endPos = lcSrcIds(lsi + 2)
        val termTopics = vattrs(si)
        useds(si) = termTopics.activeSize
        resetDist_waSparse(termDist, alphak_denoms, termTopics)
        val denseTermTopics = toBDV(termTopics)
        var pos = startPos
        while (pos < endPos) {
          val di = lcDstIds(pos)
          val docTopics = vattrs(di).asInstanceOf[Ndk]
          useds(di) = docTopics.activeSize
          val topic = data(pos)
          resetDist_dwbSparse_wAdjust(cdfDist, denoms, denseTermTopics, docTopics, topic, beta)
          val newTopic = tokenSampling(gen, global, termDist, cdfDist, denseTermTopics, topic)
          if (topic != newTopic) {
            data(pos) = newTopic
          }
          pos += 1
        }
      } finally {
        thq.add(thid + 1)
      }
    })
    withAwaitReadyAndClose(all)

    ep.withVertexAttributes(useds)
  }

  def tokenSampling(gen: Random,
    ab: DiscreteSampler[Double],
    wa: DiscreteSampler[Double],
    dwb: CumulativeDist[Double],
    denseTermTopics: BDV[Count],
    topic: Int): Int = {
    val dwbSum = dwb.norm
    val sum23 = dwbSum + wa.norm
    val distSum = sum23 + ab.norm
    val genSum = gen.nextDouble() * distSum
    if (genSum < dwbSum) {
      dwb.sampleFrom(genSum, gen)
    } else if (genSum < sum23) wa match {
      case wt: AliasTable[Double] =>
        val rr = 1.0 / denseTermTopics(topic)
        wt.resampleFrom(genSum - dwbSum, gen, topic, rr)
      case wf: FTree[Double] => wf.sampleFrom(genSum - dwbSum, gen)
    } else {
      ab.sampleFrom(genSum - sum23, gen)
    }
  }

  def resetDist_abDense(ab: FTree[Double],
    topicCounters: BDV[Count],
    alphaAS: Double,
    beta: Double,
    betaSum: Double,
    alphaRatio: Double): FTree[Double] = {
    val probs = new Array[Double](numTopics)
    var i = 0
    while (i < numTopics) {
      val nk = topicCounters(i)
      val alphak = (nk + alphaAS) * alphaRatio
      probs(i) = alphak * beta / (nk + betaSum)
      i += 1
    }
    ab.synchronized {
      ab.resetDist(probs, null, numTopics)
    }
  }
}
