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

package com.github.cloudml.zen.ml.clustering

import java.lang.ref.SoftReference
import java.util.Random
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger
import scala.concurrent._
import scala.concurrent.duration._

import LDADefines._
import com.github.cloudml.zen.ml.util._

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV}
import org.apache.spark.graphx2._
import org.apache.spark.graphx2.impl.GraphImpl


abstract class LDAAlgorithm extends Serializable {
  private[ml] def sampleGraph(corpus: Graph[TC, TA],
    topicCounters: BDV[Count],
    seed: Int,
    sampIter: Int,
    numTokens: Long,
    numTopics: Int,
    numTerms: Int,
    alpha: Double,
    alphaAS: Double,
    beta: Double): Graph[TC, TA]

  private[ml] def resampleTable[@specialized(Double, Int, Float, Long) T](gen: Random,
    table: AliasTable[T],
    counters: TC,
    topic: Int,
    cnt: Int = 0,
    numSampling: Int = 0): Int = {
    val newTopic = table.sampleRandom(gen)
    if (newTopic == topic && numSampling < 16) {
      val ntx = if (cnt == 0) counters(topic) else cnt
      // TODO: not sure it is correct or not?
      // discard it if the newly sampled topic is current topic
      if ((ntx == 1 && table.used > 1) ||
        /* the sampled topic that contains current token and other tokens */
        (ntx > 1 && gen.nextDouble() < 1.0 / ntx)
      /* the sampled topic has 1/ntx probability that belongs to current token */ ) {
        return resampleTable(gen, table, counters, topic, ntx, numSampling + 1)
      }
    }
    newTopic
  }
}

class FastLDA extends LDAAlgorithm {
  override private[ml] def sampleGraph(corpus: Graph[TC, TA],
    topicCounters: BDV[Count],
    seed: Int,
    sampIter: Int,
    numTokens: Long,
    numTopics: Int,
    numTerms: Int,
    alpha: Double,
    alphaAS: Double,
    beta: Double): GraphImpl[TC, TA] = {
    val graph = refreshEdgeAssociations(corpus)
    val edges = graph.edges
    val vertices = graph.vertices
    val conf = edges.context.getConf
    val numThreads = conf.getInt(cs_numThreads, 1)
    val sampl = conf.get(cs_accelMethod, "alias")
    val numPartitions = edges.partitions.length
    val partRDD = edges.partitionsRDD.mapPartitions(_.map(Function.tupled((pid, ep) => {
      val alphaSum = alpha * numTopics
      val alphaRatio = alphaSum / (numTokens + alphaAS * numTopics)
      val betaSum = beta * numTerms
      def itemRatio(topic: Int) = {
        val nt = topicCounters(topic)
        alphaRatio * (nt + alphaAS) / (nt + betaSum)
      }
      // table/ftree is a per term data structure
      // in GraphX, edges in a partition are clustered by source IDs (term id in this case)
      // so, use below simple cache to avoid calculating table each time
      val global: DiscreteSampler[Double] = sampl match {
        case "ftree" => new FTree[Double](numTopics, isSparse=false)
        case "alias" | "hybrid" => new AliasTable(numTopics)
      }
      tDense(global, itemRatio, beta, numTopics)
      val totalSize = ep.size
      val termSize = ep.indexSize
      val lcSrcIds = ep.localSrcIds
      val lcDstIds = ep.localDstIds
      val vattrs = ep.vertexAttrs
      val data = ep.data
      val indicator = new AtomicInteger

      implicit val es = ExecutionContext.fromExecutorService(Executors.newFixedThreadPool(numThreads))
      val all = Future.traverse(ep.index.iterator)(Function.tupled((_, offset) => Future {
        val termOrd = indicator.getAndIncrement()
        val gen = new XORShiftRandom(((seed + sampIter) * numPartitions + pid) * termSize + termOrd)
        val termDist: DiscreteSampler[Double] = sampl match {
          case "alias" => new AliasTable[Double](numTopics)
          case "ftree" | "hybrid" => new FTree(numTopics, isSparse = true)
        }
        val si = lcSrcIds(offset)
        val orgTermTopics = vattrs(si)
        wSparse(termDist, itemRatio, orgTermTopics)
        val termTopics = orgTermTopics match {
          case v: BDV[Count] => v
          case v: BSV[Count] => toBDV(v)
        }
        val cdfDist = new CumulativeDist[Double](numTopics)
        var pos = offset
        while (pos < totalSize && lcSrcIds(pos) == si) {
          val di = lcDstIds(pos)
          val docTopics = vattrs(di).asInstanceOf[BSV[Count]]
          dSparse(cdfDist, topicCounters, termTopics, docTopics, beta, betaSum)
          val topics = data(pos)
          for (i <- topics.indices) {
            val topic = topics(i)
            global.update(topic, itemRatio(topic) * beta)
            termDist.update(topic, itemRatio(topic) * termTopics(topic))
            val newTopic = tokenSampling(gen, global, termDist, cdfDist, termTopics, topic)
            topics(i) = newTopic
            global.update(newTopic, itemRatio(newTopic) * beta)
            termDist.update(newTopic, itemRatio(newTopic) * termTopics(newTopic))
          }
          pos += 1
        }
      }))
      Await.ready(all, 2.hour)
      es.shutdown()

      (pid, ep.withoutVertexAttributes[TC]().withData(data))
    })), preservesPartitioning=true)
    GraphImpl.fromExistingRDDs(vertices, edges.withPartitionsRDD(partRDD))
  }

  private[ml] def tokenSampling(gen: Random,
    t: DiscreteSampler[Double],
    w: DiscreteSampler[Double],
    d: CumulativeDist[Double],
    termTopics: BDV[Count],
    topic: Int): Int = {
    val dSum = d.norm
    val dwSum = dSum + w.norm
    val distSum = dwSum + t.norm
    val genSum = gen.nextDouble() * distSum
    if (genSum < dSum) {
      d.sampleFrom(genSum, gen)
    } else if (genSum < dwSum) {
      w match {
        case wt: AliasTable[Double] => resampleTable(gen, wt, termTopics, topic)
        case wf: FTree[Double] => wf.sampleFrom(genSum - dSum, gen)
      }
    } else {
      t.sampleFrom(genSum - dwSum, gen)
    }
  }

  /**
   * dense part in the decomposed sampling formula:
   * t = \frac{{\beta }_{w} \bar{\alpha} ( {n}_{k}^{-di} + \acute{\alpha} ) } {({n}_{k}^{-di}+\bar{\beta})
   * ({\sum{n}_{k}^{-di} +\bar{\acute{\alpha}}})}
   */
  private[ml] def tDense(t: DiscreteSampler[Double],
    itemRatio: Int => Double,
    beta: Double,
    numTopics: Int): DiscreteSampler[Double] = {
    val probs = Range(0, numTopics).map(topic => itemRatio(topic) * beta)
    t.resetDist(probs.iterator.zipWithIndex.map(_.swap), numTopics)
  }

  /**
   * word related sparse part in the decomposed sampling formula:
   * w = \frac{ {n}_{kw}^{-di} \bar{\alpha} ( {n}_{k}^{-di} + \acute{\alpha} )}{({n}_{k}^{-di}+\bar{\beta})
   * ({\sum{n}_{k}^{-di} +\bar{\acute{\alpha}}})}
   */
  private[ml] def wSparse(w: DiscreteSampler[Double],
    itemRatio: Int => Double,
    termTopics: TC): DiscreteSampler[Double] = {
    val arr = new Array[(Int, Double)](termTopics.activeSize)
    var i = 0
    for ((topic, cnt) <- termTopics.activeIterator) {
      if (cnt > 0) {
        arr(i) = (topic, cnt * itemRatio(topic))
        i += 1
      }
    }
    w.resetDist(arr.slice(0, i).iterator, i)
  }

  /**
   * doc related sparse part in the decomposed sampling formula:
   * d =  \frac{{n}_{kd} ^{-di}({\sum{n}_{k}^{-di} + \bar{\acute{\alpha}}})({n}_{kw}^{-di}+{\beta}_{w})}
   * {({n}_{k}^{-di}+\bar{\beta})({\sum{n}_{k}^{-di} +\bar{\acute{\alpha}}})}
   * =  \frac{{n}_{kd} ^{-di}({n}_{kw}^{-di}+{\beta}_{w})}{({n}_{k}^{-di}+\bar{\beta}) }
   */
  private[ml] def dSparse(d: CumulativeDist[Double],
    topicCounters: BDV[Count],
    termTopics: BDV[Count],
    docTopics: BSV[Count],
    beta: Double,
    betaSum: Double): CumulativeDist[Double] = {
    val used = docTopics.used
    val index = docTopics.index
    val data = docTopics.data
    d.directReset(i => {
      val topic = index(i)
      val cnt = data(i)
      cnt * (termTopics(topic) + beta) / (topicCounters(topic) + betaSum)
    }, used, index)
  }
}

class LightLDA extends LDAAlgorithm {
  override private[ml] def sampleGraph(corpus: Graph[TC, TA],
    topicCounters: BDV[Count],
    seed: Int,
    sampIter: Int,
    numTokens: Long,
    numTopics: Int,
    numTerms: Int,
    alpha: Double,
    alphaAS: Double,
    beta: Double): Graph[TC, TA] = {
    val graph = refreshEdgeAssociations(corpus)
    val edges = graph.edges
    val vertices = graph.vertices
    val conf = edges.context.getConf
    val numThreads = conf.getInt(cs_numThreads, 1)
    val numPartitions = edges.partitions.length
    val partRDD = edges.partitionsRDD.mapPartitions(_.map(Function.tupled((pid, ep) => {
      val alphaSum = alpha * numTopics
      val alphaRatio = alphaSum / (numTokens + alphaAS * numTopics)
      val betaSum = beta * numTerms
      val totalSize = ep.size
      val termSize = ep.indexSize
      val lcSrcIds = ep.localSrcIds
      val lcDstIds = ep.localDstIds
      val vattrs = ep.vertexAttrs
      val data = ep.data
      val vertSize = vattrs.length
      val indicator = new AtomicInteger

      val alphaDist = new AliasTable[Double](numTopics)
      val betaDist = new AliasTable[Double](numTopics)
      val docCache = new Array[SoftReference[AliasTable[Count]]](vertSize)
      dDense(alphaDist, topicCounters, numTopics, alphaRatio, alphaAS)
      wDense(betaDist, topicCounters, numTopics, beta, betaSum)
      val p = tokenTopicProb(topicCounters, beta, alpha,
        alphaAS, numTokens, numTerms) _
      val dPFun = docProb(topicCounters, alpha, alphaAS, numTokens) _
      val wPFun = wordProb(topicCounters, numTerms, beta) _

      implicit val es = ExecutionContext.fromExecutorService(Executors.newFixedThreadPool(numThreads))
      val all = Future.traverse(ep.index.iterator)(Function.tupled((_, offset) => Future {
        val termOrd = indicator.getAndIncrement()
        val gen = new XORShiftRandom(((seed + sampIter) * numPartitions + pid) * termSize + termOrd)
        val termDist = new AliasTable[Double](numTopics)
        val si = lcSrcIds(offset)
        val termTopics = vattrs(si)
        wSparse(termDist, topicCounters, termTopics, betaSum)
        var pos = offset
        while (pos < totalSize && lcSrcIds(pos) == si) {
          val di = lcDstIds(pos)
          val docTopics = vattrs(di)
          if (gen.nextDouble() < 1e-6) {
            dDense(alphaDist, topicCounters, numTopics, alphaRatio, alphaAS)
            wDense(betaDist, topicCounters, numTopics, beta, betaSum)
          }
          if (gen.nextDouble() < 1e-4) {
            wSparse(termDist, topicCounters, termTopics, betaSum)
          }
          val docDist = dSparseCached(cache => cache == null || cache.get() == null || gen.nextDouble() < 1e-2,
            docCache, docTopics, di)

          val topics = data(pos)
          for (i <- topics.indices) {
            var docProposal = gen.nextDouble() < 0.5
            for (_ <- 1 to 8) {
              docProposal = !docProposal
              val topic = topics(i)
              var proposalTopic = -1
              val q = if (docProposal) {
                val sNorm = alphaDist.norm
                val norm = sNorm + docDist.norm
                if (gen.nextDouble() * norm < sNorm) {
                  proposalTopic = alphaDist.sampleRandom(gen)
                } else {
                  proposalTopic = resampleTable(gen, docDist, docTopics, topic)
                }
                dPFun
              } else {
                val sNorm = termDist.norm
                val norm = sNorm + betaDist.norm
                val table = if (gen.nextDouble() * norm < sNorm) termDist else betaDist
                proposalTopic = table.sampleRandom(gen)
                wPFun
              }

              val newTopic = tokenSampling(gen, docTopics, termTopics, docProposal,
                topic, proposalTopic, q, p)
              if (newTopic != topic) {
                topics(i) = newTopic
                docTopics(topic) -= 1
                docTopics(newTopic) += 1
                termTopics(topic) -= 1
                termTopics(newTopic) += 1
                topicCounters(topic) -= 1
                topicCounters(newTopic) += 1
              }
            }
          }
          pos += 1
        }
      }))
      Await.ready(all, 2.hour)
      es.shutdown()

      (pid, ep.withoutVertexAttributes[TC]().withData(data))
    })), preservesPartitioning=true)
    GraphImpl.fromExistingRDDs(vertices, edges.withPartitionsRDD(partRDD))
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
  private[ml] def tokenSampling(gen: Random,
    docTopicCounter: TC,
    termTopicCounter: TC,
    docProposal: Boolean,
    currentTopic: Int,
    proposalTopic: Int,
    q: (TC, Int, Boolean) => Double,
    p: (TC, TC, Int, Boolean) => Double): Int = {
    if (proposalTopic == currentTopic) return proposalTopic
    val cp = p(docTopicCounter, termTopicCounter, currentTopic, true)
    val np = p(docTopicCounter, termTopicCounter, proposalTopic, false)
    val vd = if (docProposal) docTopicCounter else termTopicCounter
    val cq = q(vd, currentTopic, true)
    val nq = q(vd, proposalTopic, false)

    val pi = (np * cq) / (cp * nq)
    if (gen.nextDouble() < math.min(1D, pi)) proposalTopic else currentTopic
  }

  // scalastyle:off
  private[ml] def tokenTopicProb(totalTopicCounter: BDV[Count],
     beta: Double,
     alpha: Double,
     alphaAS: Double,
     numTokens: Long,
     numTerms: Int)(docTopicCounter: TC,
     termTopicCounter: TC,
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

  private[ml] def wordProb(totalTopicCounter: BDV[Count],
    numTerms: Int,
    beta: Double)(termTopicCounter: TC, topic: Int, isAdjustment: Boolean): Double = {
    (termTopicCounter(topic) + beta) / (totalTopicCounter(topic) + beta * numTerms)
  }

  private[ml] def docProb(totalTopicCounter: BDV[Count],
    alpha: Double,
    alphaAS: Double,
    numTokens: Long)(docTopicCounter: TC, topic: Int, isAdjustment: Boolean): Double = {
    val adjustment = if (isAdjustment) -1 else 0
    val numTopics = totalTopicCounter.length
    val ratio = (totalTopicCounter(topic) + alphaAS) / (numTokens - 1 + alphaAS * numTopics)
    val asPrior = ratio * (alpha * numTopics)
    docTopicCounter(topic) + adjustment + asPrior
  }

  /**
   * \frac{{\beta}_{w}}{{n}_{k}+\bar{\beta}}
   */
  private[ml] def wDense(wd: AliasTable[Double],
    topicCounters: BDV[Count],
    numTopics: Int,
    beta: Double,
    betaSum: Double): Unit = topicCounters.synchronized {
    val probs = Range(0, numTopics).map(topic => beta / (topicCounters(topic) + betaSum))
    wd.resetDist(probs.iterator.zipWithIndex.map(_.swap), numTopics)
  }

  /**
   * \frac{{n}_{kw}}{{n}_{k}+\bar{\beta}}
   */
  private[ml] def wSparse(ws: AliasTable[Double],
    topicCounters: BDV[Count],
    termTopics: TC,
    betaSum: Double): Unit = {
    val arr = new Array[(Int, Double)](termTopics.activeSize)
    var i = 0
    for ((topic, cnt) <- termTopics.activeIterator) {
      if (cnt > 0) {
        arr(i) = (topic, cnt / (topicCounters(topic) + betaSum))
        i += 1
      }
    }
    ws.resetDist(arr.slice(0, i).iterator, i)
  }

  private[ml] def dDense(dd: AliasTable[Double],
    topicCounters: BDV[Count],
    numTopics: Int,
    alphaRatio: Double,
    alphaAS: Double): Unit = topicCounters.synchronized {
    val probs = Range(0, numTopics).map(topic => alphaRatio * (topicCounters(topic) + alphaAS))
    dd.resetDist(probs.iterator.zipWithIndex.map(_.swap), numTopics)
  }

  private[ml] def dSparseCached(updatePred: SoftReference[AliasTable[Count]] => Boolean,
    cacheArray: Array[SoftReference[AliasTable[Count]]],
    docTopics: TC,
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
