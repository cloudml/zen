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

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV}
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

    val lcSrcIds = ep.localSrcIds
    val lcDstIds = ep.localDstIds
    val vattrs = ep.vertexAttrs
    val data = ep.data
    val useds = new Array[Int](vattrs.length)

    val global = new FTree[Double](isSparse=false)
    val thq = new ConcurrentLinkedQueue(1 to numThreads)
    val gens = new Array[XORShiftRandom](numThreads)
    val termDists = new Array[FTree[Double]](numThreads)
    val cdfDists = new Array[CumulativeDist[Double]](numThreads)
    resetDist_abDense(global, topicCounters, alphaAS, beta, betaSum, alphaRatio)

    implicit val es = initExecutionContext(numThreads)
    val all = Future.traverse(lcSrcIds.indices.by(3).iterator) { lsi => withFuture {
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
        resetDist_waSparse(termDist, topicCounters, termTopics, alphaAS, betaSum, alphaRatio)
        val updateCurry = updateTopicAssign(global, termDist, topicCounters, alphaAS, beta, betaSum, alphaRatio) _
        val denseTermTopics = toBDV(termTopics)
        var pos = startPos
        while (pos < endPos) {
          val di = lcDstIds(pos)
          val docTopics = vattrs(di).asInstanceOf[Ndk]
          useds(di) = docTopics.activeSize
          val tokenUpdate = updateCurry(denseTermTopics, docTopics)
          val topic = data(pos)
          tokenUpdate(topic, -1)
          resetDist_dwbSparse(cdfDist, topicCounters, denseTermTopics, docTopics, beta, betaSum)
          val newTopic = tokenSampling(gen, global, termDist, cdfDist)
          tokenUpdate(newTopic, 1)
          data(pos) = newTopic
          pos += 1
        }
      } finally {
        thq.add(thid + 1)
      }
    }}
    withAwaitReadyAndClose(all)

    ep.withVertexAttributes(useds)
  }

  def updateTopicAssign(ab: FTree[Double],
    wa: FTree[Double],
    topicCounters: BDV[Count],
    alphaAS: Double,
    beta: Double,
    betaSum: Double,
    alphaRatio: Double)
    (denseTermTopics: BDV[Count], docTopics: Ndk)
    (topic: Int, delta: Int): Unit = {
    topicCounters(topic) += delta
    denseTermTopics(topic) += delta
    docTopics.synchronized {
      docTopics(topic) += delta
    }
    val nk = topicCounters(topic)
    val denom = 1.0 / (nk + betaSum)
    val alphak = (nk + alphaAS) * alphaRatio
    ab(topic) = alphak * beta * denom
    wa(topic) = denseTermTopics(topic) * alphak * denom
  }

  def tokenSampling(gen: Random,
    ab: FTree[Double],
    wa: FTree[Double],
    dwb: CumulativeDist[Double]): Int = {
    val dwbSum = dwb.norm
    val sum23 = dwbSum + wa.norm
    val distSum = sum23 + ab.norm
    val genSum = gen.nextDouble() * distSum
    if (genSum < dwbSum) {
      dwb.sampleFrom(genSum, gen)
    } else if (genSum < sum23) {
      wa.sampleFrom(genSum - dwbSum, gen)
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
    ab.resetDist(probs, null, numTopics)
  }

  def resetDist_waSparse(wa: FTree[Double],
    topicCounters: BDV[Count],
    termTopics: Nwk,
    alphaAS: Double,
    betaSum: Double,
    alphaRatio: Double): FTree[Double] = termTopics match {
    case v: BDV[Count] =>
      val probs = new Array[Double](numTopics)
      val space = new Array[Int](numTopics)
      var psize = 0
      var i = 0
      while (i < numTopics) {
        val cnt = v(i)
        if (cnt > 0) {
          val nk = topicCounters(i)
          val alphak = (nk + alphaAS) * alphaRatio
          probs(psize) = alphak * cnt / (nk + betaSum)
          space(psize) = i
          psize += 1
        }
        i += 1
      }
      wa.resetDist(probs, space, psize)
    case v: BSV[Count] =>
      val used = v.used
      val index = v.index
      val data = v.data
      val probs = new Array[Double](used)
      var i = 0
      while (i < used) {
        val nk = topicCounters(index(i))
        val alphak = (nk + alphaAS) * alphaRatio
        probs(i) = alphak * data(i) / (nk + betaSum)
        i += 1
      }
      wa.resetDist(probs, index, used)
  }

  def resetDist_dwbSparse(dwb: CumulativeDist[Double],
    topicCounters: BDV[Count],
    denseTermTopics: BDV[Count],
    docTopics: Ndk,
    beta: Double,
    betaSum: Double): CumulativeDist[Double] = docTopics.synchronized {
    val used = docTopics.used
    val index = docTopics.index
    val data = docTopics.data
    // DANGER operations for performance
    dwb._used = used
    val cdf = dwb._cdf
    var sum = 0.0
    var i = 0
    while (i < used) {
      val topic = index(i)
      sum += (denseTermTopics(topic) + beta) * data(i) / (topicCounters(topic) + betaSum)
      cdf(i) = sum
      i += 1
    }
    dwb._space = index
    dwb
  }
}
