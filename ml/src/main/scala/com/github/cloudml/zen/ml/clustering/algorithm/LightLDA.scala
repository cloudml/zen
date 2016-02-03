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
import java.util.concurrent.ConcurrentLinkedQueue

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV}
import com.github.cloudml.zen.ml.clustering.LDADefines._
import com.github.cloudml.zen.ml.sampler._
import com.github.cloudml.zen.ml.util.Concurrent._
import com.github.cloudml.zen.ml.util.XORShiftRandom
import org.apache.spark.graphx2.impl.EdgePartition

import scala.collection.JavaConversions._
import scala.concurrent.Future


class LightLDA(numTopics: Int, numThreads: Int)
  extends LDATrainerByWord(numTopics: Int, numThreads: Int) {
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
    val vertSize = vattrs.length
    val useds = new Array[Int](vertSize)
    val thq = new ConcurrentLinkedQueue(0 until numThreads)

    val alphaDist = new AliasTable[Double]
    val betaDist = new AliasTable[Double]
    val docCache = new Array[SoftReference[AliasTable[Count]]](vertSize)
    val gens = new Array[XORShiftRandom](numThreads)
    val termDists = new Array[AliasTable[Double]](numThreads)
    val MHSamps = new Array[MetropolisHastings](numThreads)
    val compSamps = new Array[CompositeSampler](numThreads)
    resetDist_aDense(alphaDist, topicCounters, alphaAS, alphaRatio)
    resetDist_bDense(betaDist, topicCounters, beta, betaSum)
    val CGSCurry = tokenOrigProb(topicCounters, alphaAS, beta, betaSum, alphaRatio)_
    val wordPropCurry = wordProposal(topicCounters, beta, betaSum)_
    val docPropCurry = docProposal(topicCounters, alphaAS, alphaRatio)_

    implicit val es = initExecutionContext(numThreads)
    val all = Future.traverse(lcSrcIds.indices.by(3).iterator) { lsi => withFuture {
      val thid = thq.poll()
      var gen = gens(thid)
      if (gen == null) {
        gen = new XORShiftRandom(((seed + sampIter) * numPartitions + pid) * numThreads + thid)
        gens(thid) = gen
        termDists(thid) = new AliasTable[Double]
        termDists(thid).reset(numTopics)
        MHSamps(thid) = new MetropolisHastings
        compSamps(thid) = new CompositeSampler
      }
      val termDist = termDists(thid)
      val MHSamp = MHSamps(thid)
      val compSamp = compSamps(thid)

      val si = lcSrcIds(lsi)
      val startPos = lcSrcIds(lsi + 1)
      val endPos = lcSrcIds(lsi + 2)
      val termTopics = vattrs(si)
      useds(si) = termTopics.activeSize
      resetDist_wSparse(termDist, topicCounters, termTopics, betaSum)
      val wordProp = wordPropCurry(termTopics)
      var pos = startPos
      while (pos < endPos) {
        val di = lcDstIds(pos)
        val docTopics = vattrs(di).asInstanceOf[Ndk]
        useds(di) = docTopics.activeSize
        if (gen.nextDouble() < 1e-6) {
          resetDist_aDense(alphaDist, topicCounters, alphaAS, alphaRatio)
          resetDist_bDense(betaDist, topicCounters, beta, betaSum)
        }
        if (gen.nextDouble() < 1e-4) {
          resetDist_wSparse(termDist, topicCounters, termTopics, betaSum)
        }
        val docDist = dSparseCached(cache => cache == null || cache.get() == null || gen.nextDouble() < 1e-2,
          docCache, docTopics, di)

        var topic = data(pos)
        val CGSFunc = CGSCurry(termTopics, docTopics)
        val docProp = docPropCurry(docTopics)
        var docCycle = gen.nextBoolean()
        var mh = 0
        while (mh < 8) {
          if (docCycle) {
            compSamp.resetComponents(docDist, alphaDist)
            MHSamp.resetProb(CGSFunc, docProp, compSamp, topic)
          } else {
            compSamp.resetComponents(termDist, betaDist)
            MHSamp.resetProb(CGSFunc, wordProp, compSamp, topic)
          }
          val newTopic = MHSamp.sampleRandom(gen)
          if (newTopic != topic) {
            data(pos) = newTopic
            topicCounters(topic) -= 1
            topicCounters(newTopic) += 1
            termTopics(topic) -= 1
            termTopics(newTopic) += 1
            docTopics.synchronized {
              docTopics(topic) -= 1
              docTopics(newTopic) += 1
            }
            topic = newTopic
          }
          docCycle = !docCycle
          mh += 1
        }

        pos += 1
      }
      thq.add(thid)
    }}
    withAwaitReadyAndClose(all)

    ep.withVertexAttributes(useds)
  }

  def tokenOrigProb(topicCounters: BDV[Count],
    alphaAS: Double,
    beta: Double,
    betaSum: Double,
    alphaRatio: Double)
    (termTopics: Nwk, docTopics: Ndk)
    (curTopic: Int, i: Int): Double = {
    val adjust = if (i == curTopic) -1 else 0
    val ndk = docTopics.synchronized(docTopics(i))
    val alphak = (topicCounters(i) + alphaAS) * alphaRatio
    (ndk + adjust + alphak) * (termTopics(i) + adjust + beta) /
      (topicCounters(i) + adjust + betaSum)
  }

  def wordProposal(topicCounters: BDV[Count],
    beta: Double,
    betaSum: Double)
    (termTopics: Nwk)
    (curTopic: Int, i: Int): Double = {
    (termTopics(i) + beta) / (topicCounters(i) + betaSum)
  }

  def docProposal(topicCounters: BDV[Count],
    alphaAS: Double,
    alphaRatio: Double)
    (docTopics: Ndk)
    (curTopic: Int, i: Int): Double = {
    val ndk = docTopics.synchronized(docTopics(i))
    val alphak = (topicCounters(i) + alphaAS) * alphaRatio
    ndk + alphak
  }

  def resetDist_bDense(b: AliasTable[Double],
    topicCounters: BDV[Count],
    beta: Double,
    betaSum: Double): Unit = {
    val probs = new Array[Double](numTopics)
    var i = 0
    while (i < numTopics) {
      probs(i) = beta / (topicCounters(i) + betaSum)
      i += 1
    }
    b.synchronized {
      b.resetDist(probs, null, numTopics)
    }
  }

  def resetDist_wSparse(ws: AliasTable[Double],
    topicCounters: BDV[Count],
    termTopics: Nwk,
    betaSum: Double): Unit = termTopics match {
    case v: BDV[Count] =>
      val data = v.data
      val probs = new Array[Double](numTopics)
      val space = new Array[Int](numTopics)
      var psize = 0
      var i = 0
      while (i < numTopics) {
        val cnt = data(i)
        if (cnt > 0) {
          probs(psize) = cnt / (topicCounters(i) + betaSum)
          space(psize) = i
          psize += 1
        }
        i += 1
      }
      ws.resetDist(probs, space, psize)
    case v: BSV[Count] =>
      val used = v.used
      val index = v.index
      val data = v.data
      val probs = new Array[Double](used)
      var i = 0
      while (i < used) {
        probs(i) = data(i) / (topicCounters(index(i)) + betaSum)
        i += 1
      }
      ws.resetDist(probs, index, used)
  }

  def resetDist_aDense(a: AliasTable[Double],
    topicCounters: BDV[Count],
    alphaAS: Double,
    alphaRatio: Double): Unit = {
    val probs = new Array[Double](numTopics)
    var i = 0
    while (i < numTopics) {
      probs(i) = alphaRatio * (topicCounters(i) + alphaAS)
      i += 1
    }
    a.synchronized {
      a.resetDist(probs, null, numTopics)
    }
  }

  def dSparseCached(updatePred: SoftReference[AliasTable[Count]] => Boolean,
    cacheArray: Array[SoftReference[AliasTable[Count]]],
    docTopics: Ndk,
    lcDocId: Int): AliasTable[Count] = {
    val docCache = cacheArray(lcDocId)
    if (!updatePred(docCache)) {
      docCache.get
    } else {
      val tmpDocTopics = docTopics.synchronized(docTopics.copy)
      val used = tmpDocTopics.used
      val index = tmpDocTopics.index
      val data = tmpDocTopics.data
      val table = new AliasTable[Count]
      table.resetDist(data, index, used)
      cacheArray(lcDocId) = new SoftReference(table)
      table
    }
  }
}
