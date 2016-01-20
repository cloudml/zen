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


class SparseLDA(numTopics: Int, numThreads: Int)
  extends LDATrainerByDoc(numTopics: Int, numThreads: Int) {
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
    val totalSize = ep.size
    val lcSrcIds = ep.localSrcIds
    val lcDstIds = ep.localDstIds
    val vattrs = ep.vertexAttrs
    val useds = new Array[Int](vattrs.length)
    val data = ep.data
    val thq = new ConcurrentLinkedQueue(0 until numThreads)
    val gens = new Array[XORShiftRandom](numThreads)
    val docDists = new Array[FlatDist[Double]](numThreads)
    val mainDists = new Array[FlatDist[Double]](numThreads)
    val denoms = calc_denoms(topicCounters, betaSum)
    val alphak_denoms = calc_alphak_denoms(denoms, alphaAS, betaSum, alphaRatio)
    val global = new FlatDist[Double](isSparse=false)
    resetDist_abDense(global, alphak_denoms, beta)

    implicit val es = ExecutionContext.fromExecutorService(Executors.newFixedThreadPool(numThreads))
    val all = Future.traverse(ep.index.iterator)(Function.tupled((_, offset) => Future {
      val thid = thq.poll()
      var gen = gens(thid)
      if (gen == null) {
        gen = new XORShiftRandom(((seed + sampIter) * numPartitions + pid) * numThreads + thid)
        gens(thid) = gen
        docDists(thid) = new FlatDist[Double](isSparse=true)
        mainDists(thid) = new FlatDist[Double](isSparse=true)
        docDists(thid).reset(numTopics)
        mainDists(thid).reset(numTopics)
      }
      val docDist = docDists(thid)
      val si = lcSrcIds(offset)
      val docTopics = vattrs(si).asInstanceOf[BSV[Count]]
      useds(si) = docTopics.activeSize
      val nkd_denoms = calc_nkd_denoms(denoms, docTopics)
      resetDist_dbSparse(docDist, nkd_denoms, beta)
      val docAlphaK_denoms = calc_docAlphaK_denoms(alphak_denoms, nkd_denoms)
      val mainDist = mainDists(thid)
      var pos = offset
      while (pos < totalSize && lcSrcIds(pos) == si) {
        val di = lcDstIds(pos)
        val termTopics = vattrs(di)
        useds(di) = termTopics.activeSize
        resetDist_wdaSparse(mainDist, docAlphaK_denoms, termTopics)
        val topic = data(pos)
        data(pos) = tokenSampling(gen, global, docDist, mainDist)
        pos += 1
      }
      thq.add(thid)
    }))
    Await.ready(all, 2.hour)
    es.shutdown()
    ep.withVertexAttributes(useds)
  }

  def tokenSampling(gen: Random,
    ab: FlatDist[Double],
    db: FlatDist[Double],
    wda: FlatDist[Double]): Int = {
    val wdaSum = wda.norm
    val sum23 = wdaSum + db.norm
    val distSum = sum23 + ab.norm
    val genSum = gen.nextDouble() * distSum
    if (genSum < wdaSum) {
      wda.sampleFrom(genSum, gen)
    } else if (genSum < sum23) {
      db.sampleFrom(genSum - wdaSum, gen)
    } else {
      ab.sampleFrom(genSum - sum23, gen)
    }
  }
}
