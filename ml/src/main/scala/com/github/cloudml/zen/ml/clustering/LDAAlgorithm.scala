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
import java.util.concurrent.CountDownLatch

import LDADefines._
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}
import com.github.cloudml.zen.ml.util._
import org.apache.log4j.Logger
import org.apache.spark.graphx2._
import org.apache.spark.graphx2.impl.GraphImpl
import org.apache.spark.util.collection.AppendOnlyMap


abstract class LDAAlgorithm extends Serializable {
  private[ml] def sampleGraph(corpus: Graph[VD, ED],
    totalTopicCounter: BDV[Count],
    seed: Int,
    numTokens: Long,
    numTopics: Int,
    numTerms: Int,
    alpha: Double,
    alphaAS: Double,
    beta: Double): Graph[VD, ED]

  private[ml] def sampleSV(gen: Random,
    table: AliasTable[Double],
    sv: VD,
    currentTopic: Int,
    currentTopicCounter: Int = 0,
    numSampling: Int = 0): Int = {
    val docTopic = table.sampleRandom(gen)
    if (docTopic == currentTopic && numSampling < 16) {
      val svCounter = if (currentTopicCounter == 0) sv(currentTopic) else currentTopicCounter
      // TODO: not sure it is correct or not?
      // discard it if the newly sampled topic is current topic
      if ((svCounter == 1 && table.used > 1) ||
        /* the sampled topic that contains current token and other tokens */
        (svCounter > 1 && gen.nextDouble() < 1.0 / svCounter)
      /* the sampled topic has 1/svCounter probability that belongs to current token */ ) {
        return sampleSV(gen, table, sv, currentTopic, svCounter, numSampling + 1)
      }
    }
    docTopic
  }
}

class FastLDA extends LDAAlgorithm {
  override private[ml] def sampleGraph(corpus: Graph[VD, ED],
    totalTopicCounter: BDV[Count],
    seed: Int,
    numTokens: Long,
    numTopics: Int,
    numTerms: Int,
    alpha: Double,
    alphaAS: Double,
    beta: Double): Graph[VD, ED] = {
    val conf = corpus.edges.context.getConf
    val numThreads = conf.getInt(cs_numThreads, 1)
    val sampl = conf.get(cs_accelMethod, "alias")
    val graph = corpus.asInstanceOf[GraphImpl[VD, ED]]
    val vertices = graph.vertices
    val edges = graph.replicatedVertexView.edges
    val numPartitions = edges.partitions.length
    val newEdges = edges.mapEdgePartitions((pid, ep) => {
      val alphaRatio = alpha * numTopics / (numTokens - 1 + alphaAS * numTopics)
      val betaSum = beta * numTerms
      def itemRatio(topic: Int) = {
        val tCounter = totalTopicCounter(topic)
        alphaRatio * (tCounter + alphaAS) / (tCounter + betaSum)
      }
      val totalSize = ep.size
      val sizePerThrd = {
        val npt = totalSize / numThreads
        if (npt * numThreads == totalSize) npt else npt + 1
      }
      val doneSignal = new CountDownLatch(numThreads)
      val threads = new Array[Thread](numThreads)
      for (threadId <- threads.indices) {
        threads(threadId) = new Thread(new Runnable {
          override def run(): Unit = {
            val logger: Logger = Logger.getLogger(this.getClass.getName)
            val gen = new XORShiftRandom((numPartitions * seed + pid) * numThreads + threadId)
            val startPos = sizePerThrd * threadId
            val endPos = math.min(sizePerThrd * (threadId + 1), totalSize)
            val lcSrcIds = ep.localSrcIds
            val lcDstIds = ep.localSrcIds
            val vattrs = ep.vertexAttrs
            val data = ep.data
            try {
              // table/ftree is a per term data structure
              // in GraphX, edges in a partition are clustered by source IDs (term id in this case)
              // so, use below simple cache to avoid calculating table each time
              val globalSampler: DiscreteSampler[Double] = sampl match {
                case "ftree" => new FTree[Double](numTopics, isSparse=false)
                case "alias" | "hybrid" => new AliasTable(numTopics)
              }
              var lastVid: Int = -1
              val lastSampler: DiscreteSampler[Double] = sampl match {
                case "alias" => new AliasTable[Double](numTopics)
                case "ftree" | "hybrid" => new FTree(numTopics, isSparse=true)
              }
              val cdfSampler: CumulativeDist[Double] = new CumulativeDist[Double](numTopics)
              tDense(globalSampler, itemRatio, beta, numTopics)
              for (i <- startPos until endPos) {
                val localSrcId = lcSrcIds(i)
                val termTopicCounter = vattrs(localSrcId)
                if (lastVid != localSrcId) {
                  lastVid = localSrcId
                  wSparse(lastSampler, itemRatio, termTopicCounter)
                }
                val docTopicCounter = vattrs(lcDstIds(i))
                val topics = data(i)
                for (i <- topics.indices) {
                  val currentTopic = topics(i)
                  dSparse(cdfSampler, totalTopicCounter, termTopicCounter, docTopicCounter, beta, betaSum)
                  globalSampler.update(currentTopic, itemRatio(currentTopic) * beta)
                  lastSampler.update(currentTopic, itemRatio(currentTopic) * termTopicCounter(currentTopic))
                  val newTopic = tokenSampling(gen, globalSampler, lastSampler, cdfSampler, termTopicCounter,
                    docTopicCounter, currentTopic)
                  topics(i) = newTopic
                  globalSampler.update(newTopic, itemRatio(currentTopic) * beta)
                  lastSampler.update(newTopic, itemRatio(currentTopic) * termTopicCounter(currentTopic))
                }
              }
            } catch {
              case e: Exception => logger.error(e.getLocalizedMessage, e)
            } finally {
              doneSignal.countDown()
            }
          }
        }, s"sampling thread $threadId")
      }
      threads.foreach(_.start())
      doneSignal.await()
      ep
    })
    GraphImpl.fromExistingRDDs(vertices, newEdges)
  }

  private[ml] def tokenSampling(gen: Random,
    t: DiscreteSampler[Double],
    w: DiscreteSampler[Double],
    d: DiscreteSampler[Double],
    termTopicCounter: VD,
    docTopicCounter: VD,
    currentTopic: Int): Int = {
    val dSum = d.norm
    val dwSum = dSum + w.norm
    val distSum = dwSum + t.norm
    val genSum = gen.nextDouble() * distSum
    if (genSum < dSum) {
      d.sampleFrom(genSum, gen)
    } else if (genSum < dwSum) {
      w match {
        case wt: AliasTable[Double] => sampleSV(gen, wt, termTopicCounter, currentTopic)
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
    termTopicCounter: VD): DiscreteSampler[Double] = {
    val distIter = termTopicCounter.activeIterator.map {
      case (t, c) => (t, itemRatio(t) * c)
    }
    w.resetDist(distIter, termTopicCounter.used)
  }

  /**
   * doc related sparse part in the decomposed sampling formula:
   * d =  \frac{{n}_{kd} ^{-di}({\sum{n}_{k}^{-di} + \bar{\acute{\alpha}}})({n}_{kw}^{-di}+{\beta}_{w})}
   * {({n}_{k}^{-di}+\bar{\beta})({\sum{n}_{k}^{-di} +\bar{\acute{\alpha}}})}
   * =  \frac{{n}_{kd} ^{-di}({n}_{kw}^{-di}+{\beta}_{w})}{({n}_{k}^{-di}+\bar{\beta}) }
   */
  private[ml] def dSparse(d: DiscreteSampler[Double],
    totalTopicCounter: BDV[Count],
    termTopicCounter: VD,
    docTopicCounter: VD,
    beta: Double,
    betaSum: Double): DiscreteSampler[Double] = {
    val distIter = docTopicCounter.activeIterator.map {
      case (t, c) => (t, c * (termTopicCounter(t) + beta) / (totalTopicCounter(t) + betaSum))
    }
    d.resetDist(distIter, docTopicCounter.used)
  }
}

class LightLDA extends LDAAlgorithm {
  override private[ml] def sampleGraph(corpus: Graph[VD, ED],
    totalTopicCounter: BDV[Count],
    seed: Int,
    numTokens: Long,
    numTopics: Int,
    numTerms: Int,
    alpha: Double,
    alphaAS: Double,
    beta: Double): Graph[VD, ED] = {
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

      val p = tokenTopicProb(totalTopicCounter, beta, alpha,
        alphaAS, numTokens, numTerms) _
      val dPFun = docProb(totalTopicCounter, alpha, alphaAS, numTokens) _
      val wPFun = wordProb(totalTopicCounter, numTerms, beta) _

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
          var dv = dDense(totalTopicCounter, alpha, alphaAS, numTokens)
          dDSum = dv._1
          dD = AliasTable.generateAlias(dv._2)

          dv = wDense(totalTopicCounter, numTerms, beta)
          wDSum = dv._1
          wD = AliasTable.generateAlias(dv._2)
        }
        val (dSum, d) = docTopicCounter.synchronized {
          docTable(x => x == null || x.get() == null || gen.nextDouble() < 1e-2,
            docTableCache, docTopicCounter, docId)
        }
        val (wSum, w) = termTopicCounter.synchronized {
          if (lastVid != termId || gen.nextDouble() < 1e-4) {
            lastWSum = wordTable(lastTable, totalTopicCounter, termTopicCounter, termId, numTerms, beta)
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
                  sampleSV(gen, d, docTopicCounter, currentTopic)
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
            assert(newTopic >= 0 && newTopic < numTopics)
            if (newTopic != currentTopic) {
              topics(i) = newTopic
              docTopicCounter(currentTopic) -= 1
              docTopicCounter(newTopic) += 1
              termTopicCounter(currentTopic) -= 1
              termTopicCounter(newTopic) += 1
              totalTopicCounter(currentTopic) -= 1
              totalTopicCounter(newTopic) += 1
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
    docTopicCounter: VD,
    termTopicCounter: VD,
    docProposal: Boolean,
    currentTopic: Int,
    proposalTopic: Int,
    q: (VD, Int, Boolean) => Double,
    p: (VD, VD, Int, Boolean) => Double): Int = {
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
     numTerms: Int)(docTopicCounter: VD,
     termTopicCounter: VD,
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
    beta: Double)(termTopicCounter: VD, topic: Int, isAdjustment: Boolean): Double = {
    (termTopicCounter(topic) + beta) / (totalTopicCounter(topic) + beta * numTerms)
  }

  private[ml] def docProb(totalTopicCounter: BDV[Count],
    alpha: Double,
    alphaAS: Double,
    numTokens: Long)(docTopicCounter: VD, topic: Int, isAdjustment: Boolean): Double = {
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
    termTopicCounter: VD,
    numTerms: Int,
    beta: Double): (Double, BV[Double]) = {
    val numTopics = termTopicCounter.length
    val termSum = beta * numTerms
    val w = BSV.zeros[Double](numTopics)
    var sum = 0D
    val distIter = termTopicCounter.activeIterator.filter(_._2 > 0).foreach { t =>
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

  private[ml] def dSparse(docTopicCounter: VD): (Double, BV[Double]) = {
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
    docTopicCounter: VD,
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
    termTopicCounter: VD,
    termId: VertexId,
    numTerms: Int,
    beta: Double): Double = {
    val sv = wSparse(totalTopicCounter, termTopicCounter, numTerms, beta)
    table.resetDist(sv._2.activeIterator, sv._2.activeSize)
    sv._1
  }
}
