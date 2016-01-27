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
import java.util.concurrent.{ConcurrentLinkedQueue, Executors}

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV}
import com.github.cloudml.zen.ml.clustering.LDADefines._
import com.github.cloudml.zen.ml.sampler._
import com.github.cloudml.zen.ml.util.XORShiftRandom
import org.apache.spark.graphx2.impl.EdgePartition

import scala.collection.JavaConversions._
import scala.concurrent._
import scala.concurrent.duration._


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
    resetDist_aDense(alphaDist, topicCounters, numTopics, alphaRatio, alphaAS)
    resetDist_bDense(betaDist, topicCounters, numTopics, beta, betaSum)
    val p = tokenTopicProb(topicCounters, beta, alpha, alphaAS, numTokens, numTerms)_
    val dPFun = docProb(topicCounters, alpha, alphaAS, numTokens)_
    val wPFun = wordProb(topicCounters, numTerms, beta)_

    implicit val es = ExecutionContext.fromExecutorService(Executors.newFixedThreadPool(numThreads))
    val all = Future.traverse(ep.index.iterator.zipWithIndex) { case ((_, startPos), ii) => Future {
      val thid = thq.poll()
      var gen = gens(thid)
      if (gen == null) {
        gen = new XORShiftRandom(((seed + sampIter) * numPartitions + pid) * numThreads + thid)
        gens(thid) = gen
        termDists(thid) = new AliasTable[Double] { reset(numTopics) }
      }
      val termDist = termDists(thid)

      val lsi = ii << 1
      val si = lcSrcIds(lsi)
      val endPos = lcSrcIds(lsi + 1)
      val termTopics = vattrs(si)
      useds(si) = termTopics.activeSize
      resetDist_wSparse(termDist, topicCounters, termTopics, betaSum)
      var pos = startPos
      while (pos < endPos) {
        val di = lcDstIds(pos)
        val docTopics = vattrs(di).asInstanceOf[Ndk]
        useds(di) = docTopics.activeSize
        if (gen.nextDouble() < 1e-6) {
          resetDist_aDense(alphaDist, topicCounters, numTopics, alphaRatio, alphaAS)
          resetDist_bDense(betaDist, topicCounters, numTopics, beta, betaSum)
        }
        if (gen.nextDouble() < 1e-4) {
          resetDist_wSparse(termDist, topicCounters, termTopics, betaSum)
        }
        val docDist = dSparseCached(cache => cache == null || cache.get() == null || gen.nextDouble() < 1e-2,
          docCache, docTopics, di)

        val topic = data(pos)
        var docProposal = gen.nextDouble() < 0.5
        var j = 0
        while (j < 8) {
          docProposal = !docProposal
          var proposalTopic = -1
          val q = if (docProposal) {
            val aSum = alphaDist.norm
            val dPropSum = aSum + docDist.norm
            if (gen.nextDouble() * dPropSum < aSum) {
              proposalTopic = alphaDist.sampleRandom(gen)
            } else {
              val rr = 1.0 / docTopics(topic)
              proposalTopic = docDist.resampleRandom(gen, topic, rr)
            }
            dPFun
          } else {
            val wSum = termDist.norm
            val wPropSum = wSum + betaDist.norm
            val table = if (gen.nextDouble() * wPropSum < wSum) termDist else betaDist
            proposalTopic = table.sampleRandom(gen)
            wPFun
          }

          val newTopic = tokenSampling(gen, docTopics, termTopics, docProposal,
            topic, proposalTopic, q, p)
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
          }
          j += 1
        }
        pos += 1
      }
      thq.add(thid)
    }}
    Await.ready(all, 2.hour)
    es.shutdown()
    ep.withVertexAttributes(useds)
  }

  /**
    * Composition of both Gibbs sampler and Metropolis Hastings sampler
    * time complexity for each sampling is: O(1)
    * 1. sampling word-related parts of standard LDA formula via Gibbs Sampler:
    * Formula (6) in Paper "LightLDA: Big Topic Models on Modest Compute Clusters":
    * ( \frac{{n}_{kd}^{-di}+{\beta }_{w}}{{n}_{k}^{-di}+\bar{\beta }} )
    * 2. given the computed probability in step 1 as proposal distribution q in Metropolis Hasting sampling,
    * and we use asymmetric dirichlet prior, presented formula (3) in Paper "Rethinking LDA: Why Priors Matter"
    * \frac{{n}_{kw}^{-di}+{\beta }_{w}}{{n}_{k}^{-di}+\bar{\beta}} \frac{{n}_{kd} ^{-di}+ \bar{\alpha}
    * \frac{{n}_{k}^{-di} + \acute{\alpha}}{\sum{n}_{k} +\bar{\acute{\alpha}}}}{\sum{n}_{kd}^{-di} +\bar{\alpha}}
    *
    * where
    * \bar{\beta}=\sum_{w}{\beta}_{w}
    * \bar{\alpha}=\sum_{k}{\alpha}_{k}
    * \bar{\acute{\alpha}}=\bar{\acute{\alpha}}=\sum_{k}\acute{\alpha}
    * {n}_{kd} is the number of tokens in doc d that belong to topic k
    * {n}_{kw} is the number of occurrence for word w that belong to topic k
    * {n}_{k} is the number of tokens in corpus that belong to topic k
    */
  def tokenSampling(gen: Random,
    docTopicCounter: Ndk,
    termTopicCounter: Nwk,
    docProposal: Boolean,
    currentTopic: Int,
    proposalTopic: Int,
    q: (Nvk, Int, Boolean) => Double,
    p: (Ndk, Nwk, Int, Boolean) => Double): Int = {
    if (proposalTopic == currentTopic) return proposalTopic
    val cp = p(docTopicCounter, termTopicCounter, currentTopic, true)
    val np = p(docTopicCounter, termTopicCounter, proposalTopic, false)
    val vd = if (docProposal) docTopicCounter else termTopicCounter
    val cq = q(vd, currentTopic, true)
    val nq = q(vd, proposalTopic, false)

    val pi = (np * cq) / (cp * nq)
    if (gen.nextDouble() < math.min(1.0, pi)) proposalTopic else currentTopic
  }

  // scalastyle:off
  def tokenTopicProb(totalTopicCounter: BDV[Count],
    beta: Double,
    alpha: Double,
    alphaAS: Double,
    numTokens: Long,
    numTerms: Int)(docTopicCounter: Ndk,
    termTopicCounter: Nwk,
    topic: Int,
    isAdjustment: Boolean): Double = {
    val numTopics = docTopicCounter.length
    val adjustment = if (isAdjustment) -1 else 0
    val ratio = (totalTopicCounter(topic) + adjustment + alphaAS) /
      (numTokens - 1 + alphaAS * numTopics)
    val asPrior = ratio * (alpha * numTopics)
    // constant part is removed: (docLen - 1 + alpha * numTopics)
    (termTopicCounter(topic) + adjustment + beta) *
      (docTopicCounter(topic) + adjustment + asPrior) /
      (totalTopicCounter(topic) + adjustment + (numTerms * beta))

    // original form is formula (3) in Paper: "Rethinking LDA: Why Priors Matter"
    // val docLen = brzSum(docTopicCounter)
    // (termTopicCounter(topic) + adjustment + beta) * (docTopicCounter(topic) + adjustment + asPrior) /
    //   ((totalTopicCounter(topic) + adjustment + (numTerms * beta)) * (docLen - 1 + alpha * numTopics))
  }

  // scalastyle:on

  def wordProb(totalTopicCounter: BDV[Count],
    numTerms: Int,
    beta: Double)(termTopicCounter: Nvk, topic: Int, isAdjustment: Boolean): Double = {
    (termTopicCounter(topic) + beta) / (totalTopicCounter(topic) + beta * numTerms)
  }

  def docProb(totalTopicCounter: BDV[Count],
    alpha: Double,
    alphaAS: Double,
    numTokens: Long)(docTopicCounter: Nvk, topic: Int, isAdjustment: Boolean): Double = {
    val adjustment = if (isAdjustment) -1 else 0
    val numTopics = totalTopicCounter.length
    val ratio = (totalTopicCounter(topic) + alphaAS) / (numTokens - 1 + alphaAS * numTopics)
    val asPrior = ratio * (alpha * numTopics)
    docTopicCounter(topic) + adjustment + asPrior
  }

  /**
    * \frac{{\beta}_{w}}{{n}_{k}+\bar{\beta}}
    */
  def resetDist_bDense(b: AliasTable[Double],
    topicCounters: BDV[Count],
    numTopics: Int,
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

  /**
    * \frac{{n}_{kw}}{{n}_{k}+\bar{\beta}}
    */
  def resetDist_wSparse(ws: AliasTable[Double],
    topicCounters: BDV[Count],
    termTopics: Nwk,
    betaSum: Double): Unit = termTopics match {
    case v: BDV[Count] =>
      val numTopics = v.length
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
    numTopics: Int,
    alphaRatio: Double,
    alphaAS: Double): Unit = {
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
      val table = AliasTable.generateAlias(docTopics)
      cacheArray(lcDocId) = new SoftReference(table)
      table
    }
  }
}
