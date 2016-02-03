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

import java.lang.ref.SoftReference
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


class AliasLDA(numTopics: Int, numThreads: Int)
  extends LDATrainerByDoc(numTopics: Int, numThreads: Int) {
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

    val totalSize = ep.size
    val lcSrcIds = ep.localSrcIds
    val lcDstIds = ep.localDstIds
    val vattrs = ep.vertexAttrs
    val data = ep.data
    val vertSize = vattrs.length
    val useds = new Array[Int](vertSize)
    val termCache = new Array[SoftReference[AliasTable[Double]]](vertSize)

    val global = new AliasTable[Double]
    val thq = new ConcurrentLinkedQueue(0 until numThreads)
    val gens = new Array[XORShiftRandom](numThreads)
    val docDists = new Array[AliasTable[Double]](numThreads)
    val MHSamps = new Array[MetropolisHastings](numThreads)
    val compSamps = new Array[CompositeSampler](numThreads)
    resetDist_abDense(global, topicCounters, alphaAS, beta, betaSum, alphaRatio)

    implicit val es = initExecutionContext(numThreads)
    val all = Future.traverse(ep.index.iterator)(Function.tupled((_, offset) => withFuture {
      val thid = thq.poll()
      var gen = gens(thid)
      if (gen == null) {
        gen = new XORShiftRandom(((seed + sampIter) * numPartitions + pid) * numThreads + thid)
        gens(thid) = gen
        docDists(thid) = new AliasTable[Double].reset(numTopics)
        MHSamps(thid) = new MetropolisHastings
        compSamps(thid) = new CompositeSampler
      }
      val docDist = docDists(thid)
      val MHSamp = MHSamps(thid)
      val compSamp = compSamps(thid)

      val si = lcSrcIds(offset)
      val docTopics = vattrs(si).asInstanceOf[BSV[Count]]
      useds(si) = docTopics.activeSize
      var pos = offset
      while (pos < totalSize && lcSrcIds(pos) == si) {
        val di = lcDstIds(pos)
        val termTopics = vattrs(di)
        useds(di) = termTopics.activeSize
        val termDist = waSparseCached(termCache, di, gen.nextDouble() < 1e-2).getOrElse {
          resetCache_waSparse(termCache, di, topicCounters, termTopics, alphaAS, betaSum, alphaRatio)
        }
        resetDist_dwbSparse(docDist, topicCounters, termTopics, docTopics, beta, betaSum)
        val topic = data(pos)
        compSamp.resetComponents(docDist, termDist, global)
        // MHSamp.resetProb(topic)
        // data(pos) = tokenSampling(gen, global, docDist, mainDist)
        pos += 1
      }
      thq.add(thid)
    }))
    withAwaitReadyAndClose(all)

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

  def resetDist_abDense(ab: AliasTable[Double],
    topicCounters: BDV[Count],
    alphaAS: Double,
    beta: Double,
    betaSum: Double,
    alphaRatio: Double): DiscreteSampler[Double] = {
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

  def resetDist_dwbSparse(dwb: AliasTable[Double],
    topicCounter: BDV[Count],
    termTopics: Nwk,
    docTopics: Ndk,
    beta: Double,
    betaSum: Double): AliasTable[Double] = {
    val used = docTopics.used
    val index = docTopics.index
    val data = docTopics.data
    val probs = new Array[Double](used)
    var i = 0
    while (i < used) {
      val topic = index(i)
      probs(i) = (termTopics(topic) + beta) * data(i) / (topicCounter(topic) + betaSum)
      i += 1
    }
    dwb.resetDist(probs, index, used)
  }

  def waSparseCached(cache: Array[SoftReference[AliasTable[Double]]],
    ci: Int,
    needRefresh: => Boolean): Option[AliasTable[Double]] = {
    val termCache = cache(ci)
    if (termCache == null || termCache.get == null || needRefresh) {
      None
    } else {
      Some(termCache.get)
    }
  }

  def resetCache_waSparse(cache: Array[SoftReference[AliasTable[Double]]],
    ci: Int,
    topicCounters: BDV[Count],
    termTopics: Nwk,
    alphaAS: Double,
    betaSum: Double,
    alphaRatio: Double): AliasTable[Double] = {
    val tmpDocTopics = termTopics.synchronized(termTopics.copy)
    val table = new AliasTable[Double]
    tmpDocTopics match {
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
        table.resetDist(probs, space, psize)
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
        table.resetDist(probs, index, used)
    }
    cache(ci) = new SoftReference(table)
    table
  }
}
