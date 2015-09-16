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
import scala.concurrent.duration.Duration

import LDADefines._
import com.github.cloudml.zen.ml.util._

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}
import org.apache.spark.graphx2._
import org.apache.spark.graphx2.impl.GraphImpl
import org.apache.spark.util.collection.AppendOnlyMap


abstract class LDAAlgorithm extends Serializable {
  private[ml] def sampleGraph(corpus: Graph[TC, TA],
    topicCounters: BDV[Count],
    seed: Int,
    numTokens: Long,
    numTopics: Int,
    numTerms: Int,
    alpha: Double,
    alphaAS: Double,
    beta: Double): Graph[TC, TA]

  private[ml] def resampleTable(gen: Random,
    table: AliasTable[Double],
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
    numTokens: Long,
    numTopics: Int,
    numTerms: Int,
    alpha: Double,
    alphaAS: Double,
    beta: Double): GraphImpl[TC, TA] = {
    val graph = refreshEdgeAssociations(corpus)
    val vertices = graph.vertices
    val edges = graph.edges
    val conf = edges.context.getConf
    val numThreads = conf.getInt(cs_numThreads, 1)
    val sampl = conf.get(cs_accelMethod, "alias")
    val numPartitions = edges.partitions.length
    val newEdges = edges.mapEdgePartitions((pid, ep) => {
      val alphaRatio = alpha * numTopics / (numTokens - 1 + alphaAS * numTopics)
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

      implicit val ec = ExecutionContext.fromExecutorService(Executors.newFixedThreadPool(numThreads))
      val all = Future.traverse(ep.index.iterator)(t => Future {
        val termId = indicator.getAndIncrement()
        val gen = new XORShiftRandom((seed * numPartitions + pid) * termSize + termId)
        val wordDist: DiscreteSampler[Double] = sampl match {
          case "alias" => new AliasTable[Double](numTopics)
          case "ftree" | "hybrid" => new FTree(numTopics, isSparse = true)
        }
        var pos = t._2
        val si = lcSrcIds(pos)
        val orgTermTopics = vattrs(si)
        wSparse(wordDist, itemRatio, orgTermTopics)
        val termTopics = orgTermTopics match {
          case v: BDV[Count] => v
          case v: BSV[Count] => toBDV(v)
        }
        val cdfDist = new CumulativeDist[Double](numTopics)
        while (pos < totalSize && lcSrcIds(pos) == si) {
          val docTopics = vattrs(lcDstIds(pos))
          dSparse(cdfDist, topicCounters, termTopics, docTopics, beta, betaSum)
          val topics = data(pos)
          for (i <- topics.indices) {
            val topic = topics(i)
            global.update(topic, itemRatio(topic) * beta)
            wordDist.update(topic, itemRatio(topic) * termTopics(topic))
            val newTopic = tokenSampling(gen, global, wordDist, cdfDist, termTopics, topic)
            topics(i) = newTopic
            global.update(newTopic, itemRatio(newTopic) * beta)
            wordDist.update(newTopic, itemRatio(newTopic) * termTopics(newTopic))
          }
          pos += 1
        }
      })
      Await.ready(all, Duration.Inf)
      ec.shutdown()

      ep.withoutVertexAttributes[TC]().withData(data)
    })
    GraphImpl.fromExistingRDDs(vertices, newEdges)
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
    val dist = Range(0, numTopics).map(t => (t, itemRatio(t) * beta))
    t.resetDist(dist.iterator, dist.length)
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
    docTopics: TC,
    beta: Double,
    betaSum: Double): CumulativeDist[Double] = {
    val dtc = docTopics.asInstanceOf[BSV[Count]]
    val used = dtc.used
    val index = dtc.index
    val data = dtc.data
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
    numTokens: Long,
    numTopics: Int,
    numTerms: Int,
    alpha: Double,
    alphaAS: Double,
    beta: Double): Graph[TC, TA] = {
    val numPartitions = corpus.edges.partitions.length
    val sampledCorpus = corpus.mapTriplets((pid, iter) => {
      val gen = new XORShiftRandom(numPartitions * seed + pid)
      val docTableCache = new AppendOnlyMap[VertexId, SoftReference[(Double, AliasTable[Double])]]()

      // table is a per term data structure
      // in GraphX, edges in a partition are clustered by source IDs (term id in this case)
      // so, use below simple cache to avoid calculating table each time
      val lastTable = new AliasTable[Double](numTopics.toInt)
      var lastVid: VertexId = -1L
      var lastWSum = 0D

      val p = tokenTopicProb(topicCounters, beta, alpha,
        alphaAS, numTokens, numTerms) _
      val dPFun = docProb(topicCounters, alpha, alphaAS, numTokens) _
      val wPFun = wordProb(topicCounters, numTerms, beta) _

      var dD: AliasTable[Double] = null
      var dDSum = 0D
      var wD: AliasTable[Double] = null
      var wDSum = 0D

      iter.map(triplet => {
        val termId = triplet.srcId
        val docId = triplet.dstId
        val termTopicCounter = triplet.srcAttr
        val docTopicCounter = triplet.dstAttr
        val topics = triplet.attr

        if (dD == null || gen.nextDouble() < 1e-6) {
          var dv = dDense(topicCounters, alpha, alphaAS, numTokens)
          dDSum = dv._1
          dD = AliasTable.generateAlias(dv._2)

          dv = wDense(topicCounters, numTerms, beta)
          wDSum = dv._1
          wD = AliasTable.generateAlias(dv._2)
        }
        val (dSum, d) = docTopicCounter.synchronized {
          docTable(x => x == null || x.get() == null || gen.nextDouble() < 1e-2,
            docTableCache, docTopicCounter, docId)
        }
        val (wSum, w) = termTopicCounter.synchronized {
          if (lastVid != termId || gen.nextDouble() < 1e-4) {
            lastWSum = wordTable(lastTable, topicCounters, termTopicCounter, termId, numTerms, beta)
            lastVid = termId
          }
          (lastWSum, lastTable)
        }
        for (i <- topics.indices) {
          var docProposal = gen.nextDouble() < 0.5
          var maxSampling = 8
          while (maxSampling > 0) {
            maxSampling -= 1
            docProposal = !docProposal
            val currentTopic = topics(i)
            var proposalTopic = -1
            val q = if (docProposal) {
              if (gen.nextDouble() < dDSum / (dSum - 1 + dDSum)) {
                proposalTopic = dD.sampleRandom(gen)
              }
              else {
                proposalTopic = docTopicCounter.synchronized {
                  resampleTable(gen, d, docTopicCounter, currentTopic)
                }
              }
              dPFun
            } else {
              val table = if (gen.nextDouble() < wSum / (wSum + wDSum)) w else wD
              proposalTopic = table.sampleRandom(gen)
              wPFun
            }

            val newTopic = tokenSampling(gen, docTopicCounter, termTopicCounter, docProposal,
              currentTopic, proposalTopic, q, p)
            if (newTopic != currentTopic) {
              topics(i) = newTopic
              docTopicCounter(currentTopic) -= 1
              docTopicCounter(newTopic) += 1
              termTopicCounter(currentTopic) -= 1
              termTopicCounter(newTopic) += 1
              topicCounters(currentTopic) -= 1
              topicCounters(newTopic) += 1
            }
          }
        }
        topics
      })
    }, TripletFields.All)
    GraphImpl(sampledCorpus.vertices.mapValues(_ => null), sampledCorpus.edges)
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
   * \frac{{n}_{kw}}{{n}_{k}+\bar{\beta}}
   */
  private[ml] def wSparse(totalTopicCounter: BDV[Count],
    termTopicCounter: TC,
    numTerms: Int,
    beta: Double): (Double, BV[Double]) = {
    val numTopics = termTopicCounter.length
    val termSum = beta * numTerms
    val w = BSV.zeros[Double](numTopics)
    var sum = 0D
    termTopicCounter.activeIterator.filter(_._2 > 0).foreach { t =>
      val topic = t._1
      val count = t._2
      if (count > 0) {
        val last = count / (totalTopicCounter(topic) + termSum)
        w(topic) = last
        sum += last
      }
    }
    (sum, w)
  }

  /**
   * \frac{{\beta}_{w}}{{n}_{k}+\bar{\beta}}
   */
  private[ml] def wDense(totalTopicCounter: BDV[Count],
    numTerms: Int,
    beta: Double): (Double, BV[Double]) = {
    val numTopics = totalTopicCounter.length
    val t = BDV.zeros[Double](numTopics)
    val termSum = beta * numTerms
    var sum = 0D
    for (topic <- 0 until numTopics) {
      val last = beta / (totalTopicCounter(topic) + termSum)
      t(topic) = last
      sum += last
    }
    (sum, t)
  }

  private[ml] def dSparse(docTopicCounter: TC): (Double, BV[Double]) = {
    val numTopics = docTopicCounter.length
    val d = BSV.zeros[Double](numTopics)
    var sum = 0D
    docTopicCounter.activeIterator.foreach { t =>
      val topic = t._1
      val count = t._2
      if (count > 0) {
        val last = count
        d(topic) = last
        sum += last
      }
    }
    (sum, d)
  }

  private[ml] def dDense(totalTopicCounter: BDV[Count],
    alpha: Double,
    alphaAS: Double,
    numTokens: Long): (Double, BV[Double]) = {
    val numTopics = totalTopicCounter.length
    val asPrior = BDV.zeros[Double](numTopics)
    var sum = 0D
    for (topic <- 0 until numTopics) {
      val ratio = (totalTopicCounter(topic) + alphaAS) /
        (numTokens - 1 + alphaAS * numTopics)
      val last = ratio * (alpha * numTopics)
      asPrior(topic) = last
      sum += last
    }
    (sum, asPrior)
  }

  private[ml] def docTable(updateFunc: SoftReference[(Double, AliasTable[Double])] => Boolean,
    cacheMap: AppendOnlyMap[VertexId, SoftReference[(Double, AliasTable[Double])]],
    docTopicCounter: TC,
    docId: VertexId): (Double, AliasTable[Double]) = {
    val cacheD = cacheMap(docId)
    if (!updateFunc(cacheD)) {
      cacheD.get
    } else {
      docTopicCounter.synchronized {
        val sv = dSparse(docTopicCounter)
        val d = (sv._1, AliasTable.generateAlias(sv._2))
        cacheMap.update(docId, new SoftReference(d))
        d
      }
    }
  }

  private[ml] def wordTable(table: AliasTable[Double],
    totalTopicCounter: BDV[Count],
    termTopicCounter: TC,
    termId: VertexId,
    numTerms: Int,
    beta: Double): Double = {
    val sv = wSparse(totalTopicCounter, termTopicCounter, numTerms, beta)
    table.resetDist(sv._2.activeIterator, sv._2.activeSize)
    sv._1
  }
}
