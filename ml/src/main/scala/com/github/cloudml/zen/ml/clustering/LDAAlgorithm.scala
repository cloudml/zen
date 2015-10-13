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
import java.util.concurrent.atomic.AtomicIntegerArray
import java.util.concurrent.{ConcurrentLinkedQueue, Executors}
import scala.collection.JavaConversions._
import scala.concurrent._
import scala.concurrent.duration._

import LDADefines._
import com.github.cloudml.zen.ml.util._

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}
import org.apache.spark.graphx2._
import org.apache.spark.graphx2.impl.GraphImpl
import spire.math.{Numeric => spNum}


abstract class LDAAlgorithm extends Serializable {
  def sampleGraph(corpus: Graph[TC, TA],
    topicCounters: BDV[Count],
    seed: Int,
    sampIter: Int,
    numTokens: Long,
    numTopics: Int,
    numTerms: Int,
    alpha: Double,
    alphaAS: Double,
    beta: Double): GraphImpl[TC, TA]

  def updateVertexCounters(newCorpus: Graph[TC, TA],
    numTopics: Int,
    inferenceOnly: Boolean = false): GraphImpl[TC, TA]

  def resampleTable[@specialized(Double, Int, Float, Long) T: spNum](gen: Random,
    table: AliasTable[T],
    state: Int,
    cnt: Int = 0,
    numSampling: Int = 0): Int = {
    val newState = table.sampleRandom(gen)
    if (newState == state && numSampling < 16) {
      // discard it if the newly sampled topic is current topic
      if ((cnt == 1 && table.used > 1) ||
        /* the sampled topic that contains current token and other tokens */
        (cnt > 1 && gen.nextDouble() < 1.0 / cnt)
      /* the sampled topic has 1/cnt probability that belongs to current token */ ) {
        return resampleTable(gen, table, state, cnt, numSampling + 1)
      }
    }
    newState
  }
}

abstract class LDAWordByWord extends LDAAlgorithm {
  override def updateVertexCounters(sampledCorpus: Graph[TC, TA],
    numTopics: Int,
    inferenceOnly: Boolean = false): GraphImpl[TC, TA] = {
    val dscp = numTopics >> 3
    val graph = sampledCorpus.asInstanceOf[GraphImpl[TC, TA]]
    val vertices = graph.vertices
    val edges = graph.edges
    val conf = edges.context.getConf
    val numThreads = conf.getInt(cs_numThreads, 1)
    val shippedCounters = edges.partitionsRDD.mapPartitions(_.flatMap(Function.tupled((_, ep) => {
      val totalSize = ep.size
      val lcSrcIds = ep.localSrcIds
      val lcDstIds = ep.localDstIds
      val l2g = ep.local2global
      val vattrs = ep.vertexAttrs
      val data = ep.data
      val vertSize = vattrs.length
      val results = new Array[(VertexId, TC)](vertSize)
      val marks = new AtomicIntegerArray(vertSize)

      implicit val es = ExecutionContext.fromExecutorService(Executors.newFixedThreadPool(numThreads))
      val all = Future.traverse(ep.index.iterator)(Function.tupled((_, offset) => Future {
        val si = lcSrcIds(offset)
        var termTuple = results(si)
        if (termTuple == null && !inferenceOnly) {
          termTuple = (l2g(si), BSV.zeros[Count](numTopics))
          results(si) = termTuple
        }
        var termTopics = if (!inferenceOnly) termTuple._2 else null
        var pos = offset
        while (pos < totalSize && lcSrcIds(pos) == si) {
          val di = lcDstIds(pos)
          var docTuple = results(di)
          if (docTuple == null) {
            if (marks.getAndDecrement(di) == 0) {
              docTuple = (l2g(di), BSV.zeros[Count](numTopics))
              results(di) = docTuple
              marks.set(di, Int.MaxValue)
            } else {
              while (marks.get(di) <= 0) {}
              docTuple = results(di)
            }
          }
          val docTopics = docTuple._2
          val topics = data(pos)
          var i = 0
          while (i < topics.length) {
            val topic = topics(i)
            if (!inferenceOnly) termTopics match {
              case v: BDV[Count] => v(topic) += 1
              case v: BSV[Count] =>
                v(topic) += 1
                if (v.activeSize >= dscp) {
                  termTuple = (l2g(si), toBDV(v))
                  results(si) = termTuple
                  termTopics = termTuple._2
                }
            }
            docTopics.synchronized {
              docTopics(topic) += 1
            }
            i += 1
          }
          pos += 1
        }
      }))
      Await.ready(all, 1.hour)
      es.shutdown()

      results.iterator.filter(_ != null)
    }))).partitionBy(vertices.partitioner.get)

    val partRDD = vertices.partitionsRDD.zipPartitions(shippedCounters, preservesPartitioning=true)(
      (svpIter, cntsIter) => svpIter.map(svp => {
        val results = svp.values
        val index = svp.index
        val marks = new AtomicIntegerArray(results.length)
        implicit val es = ExecutionContext.fromExecutorService(Executors.newFixedThreadPool(numThreads))
        val all = cntsIter.grouped(numThreads * 5).map(batch => Future {
          batch.foreach(Function.tupled((vid, counter) => {
            val i = index.getPos(vid)
            if (marks.getAndDecrement(i) == 0) {
              results(i) = counter
            } else {
              while (marks.getAndSet(i, -1) <= 0) {}
              val agg = results(i)
              results(i) = if (isTermId(vid)) agg match {
                case u: BDV[Count] => counter match {
                  case v: BDV[Count] => u :+= v
                  case v: BSV[Count] => u :+= v
                }
                case u: BSV[Count] => counter match {
                  case v: BDV[Count] => v :+= u
                  case v: BSV[Count] =>
                    u :+= v
                    if (u.activeSize >= dscp) toBDV(u) else u
                }
              } else agg match {
                case u: BSV[Count] => counter match {
                  case v: BSV[Count] => u :+= v
                }
              }
            }
            marks.set(i, Int.MaxValue)
          }))
        })
        Await.ready(Future.sequence(all), 1.hour)
        es.shutdown()
        svp.withValues(results)
      })
    )
    GraphImpl.fromExistingRDDs(vertices.withPartitionsRDD(partRDD), edges)
  }
}

class FastLDA extends LDAWordByWord {
  override def sampleGraph(corpus: Graph[TC, TA],
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
      val alphaRatio = alpha * numTopics / (numTokens + alphaAS * numTopics)
      val betaSum = beta * numTerms
      val denoms = calcDenoms(topicCounters, numTopics, betaSum)
      val alphaK_denoms = (denoms.copy :*= ((alphaAS - betaSum) * alphaRatio)) :+= alphaRatio
      val beta_denoms = denoms.copy :*= beta
      val totalSize = ep.size
      val lcSrcIds = ep.localSrcIds
      val lcDstIds = ep.localDstIds
      val vattrs = ep.vertexAttrs
      val data = ep.data
      val thq = new ConcurrentLinkedQueue(0 until numThreads)
      // table/ftree is a per term data structure
      // in GraphX, edges in a partition are clustered by source IDs (term id in this case)
      // so, use below simple cache to avoid calculating table each time
      val global: DiscreteSampler[Double] = sampl match {
        case "ftree" => new FTree[Double](numTopics, isSparse=false)
        case "alias" | "hybrid" => new AliasTable(numTopics)
      }
      val gens = new Array[XORShiftRandom](numThreads)
      val termDists = new Array[DiscreteSampler[Double]](numThreads)
      val cdfDists = new Array[CumulativeDist[Double]](numThreads)
      tDense(global, alphaK_denoms, beta, numTopics)

      implicit val es = ExecutionContext.fromExecutorService(Executors.newFixedThreadPool(numThreads))
      val all = Future.traverse(ep.index.iterator)(Function.tupled((_, offset) => Future {
        val thid = thq.poll()
        var gen = gens(thid)
        if (gen == null) {
          gen = new XORShiftRandom(((seed + sampIter) * numPartitions + pid) * numThreads + thid)
          gens(thid) = gen
          termDists(thid) = sampl match {
            case "alias" => new AliasTable[Double](numTopics)
            case "ftree" | "hybrid" => new FTree(numTopics, isSparse=true)
          }
          cdfDists(thid) = new CumulativeDist[Double](numTopics)
        }
        val termDist = termDists(thid)
        val si = lcSrcIds(offset)
        val orgTermTopics = vattrs(si)
        wSparse(termDist, alphaK_denoms, orgTermTopics)
        val termTopics = orgTermTopics match {
          case v: BDV[Count] => v
          case v: BSV[Count] => toBDV(v)
        }
        val termBeta_denoms = calcTermBetaDenoms(orgTermTopics, denoms, beta_denoms, numTopics)
        val cdfDist = cdfDists(thid)
        var pos = offset
        while (pos < totalSize && lcSrcIds(pos) == si) {
          val di = lcDstIds(pos)
          val docTopics = vattrs(di).asInstanceOf[BSV[Count]]
          dSparse(cdfDist, docTopics, termBeta_denoms)
          val topics = data(pos)
          var i = 0
          while (i < topics.length) {
            val topic = topics(i)
            val newTopic = tokenSampling(gen, global, termDist, cdfDist, termTopics, topic)
            topics(i) = newTopic
            i += 1
          }
          pos += 1
        }
        thq.add(thid)
      }))
      Await.ready(all, 2.hour)
      es.shutdown()

      (pid, ep.withoutVertexAttributes[TC]())
    })), preservesPartitioning=true)
    GraphImpl.fromExistingRDDs(vertices, edges.withPartitionsRDD(partRDD))
  }

  def tokenSampling(gen: Random,
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
    } else if (genSum < dwSum) w match {
      case wt: AliasTable[Double] => resampleTable(gen, wt, topic, termTopics(topic))
      case wf: FTree[Double] => wf.sampleFrom(genSum - dSum, gen)
    } else {
      t.sampleFrom(genSum - dwSum, gen)
    }
  }

  /**
   * dense part in the decomposed sampling formula:
   * t = \frac{{\beta }_{w} \bar{\alpha} ( {n}_{k}^{-di} + \acute{\alpha} ) } {({n}_{k}^{-di}+\bar{\beta})
   * ({\sum{n}_{k}^{-di} +\bar{\acute{\alpha}}})}
   */
  def tDense(t: DiscreteSampler[Double],
    alphaK_denoms: BDV[Double],
    beta: Double,
    numTopics: Int): DiscreteSampler[Double] = {
    val probs = alphaK_denoms.copy :*= beta
    t.resetDist(probs.data, null, numTopics)
  }

  /**
   * word related sparse part in the decomposed sampling formula:
   * w = \frac{ {n}_{kw}^{-di} \bar{\alpha} ( {n}_{k}^{-di} + \acute{\alpha} )}{({n}_{k}^{-di}+\bar{\beta})
   * ({\sum{n}_{k}^{-di} +\bar{\acute{\alpha}}})}
   */
  def wSparse(w: DiscreteSampler[Double],
    alphaK_denoms: BDV[Double],
    termTopics: TC): DiscreteSampler[Double] = termTopics match {
    case v: BDV[Count] =>
      val numTopics = v.length
      val probs = new Array[Double](numTopics)
      val space = new Array[Int](numTopics)
      var psize = 0
      var i = 0
      while (i < numTopics) {
        val cnt = v(i)
        if (cnt > 0) {
          probs(psize) = alphaK_denoms(i) * cnt
          space(psize) = i
          psize += 1
        }
        i += 1
      }
      w.resetDist(probs, space, psize)
    case v: BSV[Count] =>
      val used = v.used
      val index = v.index
      val data = v.data
      val probs = new Array[Double](used)
      var i = 0
      while (i < used) {
        probs(i) = alphaK_denoms(index(i)) * data(i)
        i += 1
      }
      w.resetDist(probs, index, used)
  }

  /**
   * doc related sparse part in the decomposed sampling formula:
   * d =  \frac{{n}_{kd} ^{-di}({\sum{n}_{k}^{-di} + \bar{\acute{\alpha}}})({n}_{kw}^{-di}+{\beta}_{w})}
   * {({n}_{k}^{-di}+\bar{\beta})({\sum{n}_{k}^{-di} +\bar{\acute{\alpha}}})}
   * =  \frac{{n}_{kd} ^{-di}({n}_{kw}^{-di}+{\beta}_{w})}{({n}_{k}^{-di}+\bar{\beta}) }
   */
  def dSparse(d: CumulativeDist[Double],
    docTopics: BSV[Count],
    termBeta_denoms: BDV[Double]): CumulativeDist[Double] = {
    val used = docTopics.used
    val index = docTopics.index
    val data = docTopics.data
    // DANGER operations for performance
    d._used = used
    val cdf = d._cdf
    var sum = 0.0
    var i = 0
    while (i < used) {
      sum += data(i) * termBeta_denoms(index(i))
      cdf(i) = sum
      i += 1
    }
    d._space = index
    d
  }

  def calcDenoms(topicCounters: BDV[Count],
    numTopics: Int,
    betaSum: Double): BDV[Double] = {
    val bdv = BDV.zeros[Double](numTopics)
    var i = 0
    while (i < numTopics) {
      bdv(i) = 1.0 / (topicCounters(i) + betaSum)
      i += 1
    }
    bdv
  }

  def calcTermBetaDenoms(orgTermTopics: BV[Count],
    denoms: BDV[Double],
    beta_denoms: BDV[Double],
    numTopics: Int): BDV[Double] = {
    val bdv = beta_denoms.copy
    orgTermTopics match {
      case v: BDV[Count] =>
        var i = 0
        while (i < numTopics) {
          bdv(i) += v(i) * denoms(i)
          i += 1
        }
      case v: BSV[Count] =>
        val used = v.used
        val index = v.index
        val data = v.data
        var i = 0
        while (i < used) {
          val topic = index(i)
          bdv(topic) += data(i) * denoms(topic)
          i += 1
        }
    }
    bdv
  }
}

class LightLDA extends LDAWordByWord {
  override def sampleGraph(corpus: Graph[TC, TA],
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
    val numPartitions = edges.partitions.length
    val partRDD = edges.partitionsRDD.mapPartitions(_.map(Function.tupled((pid, ep) => {
      val alphaSum = alpha * numTopics
      val alphaRatio = alphaSum / (numTokens + alphaAS * numTopics)
      val betaSum = beta * numTerms
      val totalSize = ep.size
      val lcSrcIds = ep.localSrcIds
      val lcDstIds = ep.localDstIds
      val vattrs = ep.vertexAttrs
      val data = ep.data
      val vertSize = vattrs.length
      val thq = new ConcurrentLinkedQueue(0 until numThreads)

      val alphaDist = new AliasTable[Double](numTopics)
      val betaDist = new AliasTable[Double](numTopics)
      val docCache = new Array[SoftReference[AliasTable[Count]]](vertSize)
      val gens = new Array[XORShiftRandom](numThreads)
      val termDists = new Array[AliasTable[Double]](numThreads)
      dDense(alphaDist, topicCounters, numTopics, alphaRatio, alphaAS)
      wDense(betaDist, topicCounters, numTopics, beta, betaSum)
      val p = tokenTopicProb(topicCounters, beta, alpha,
        alphaAS, numTokens, numTerms) _
      val dPFun = docProb(topicCounters, alpha, alphaAS, numTokens) _
      val wPFun = wordProb(topicCounters, numTerms, beta) _

      implicit val es = ExecutionContext.fromExecutorService(Executors.newFixedThreadPool(numThreads))
      val all = Future.traverse(ep.index.iterator)(Function.tupled((_, offset) => Future {
        val thid = thq.poll()
        var gen = gens(thid)
        if (gen == null) {
          gen = new XORShiftRandom(((seed + sampIter) * numPartitions + pid) * numThreads + thid)
          gens(thid) = gen
          termDists(thid) = new AliasTable[Double](numTopics)
        }
        val termDist = termDists(thid)
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
          var i = 0
          while (i < topics.length) {
            var docProposal = gen.nextDouble() < 0.5
            var j = 0
            while (j < 8) {
              docProposal = !docProposal
              val topic = topics(i)
              var proposalTopic = -1
              val q = if (docProposal) {
                val sNorm = alphaDist.norm
                val norm = sNorm + docDist.norm
                if (gen.nextDouble() * norm < sNorm) {
                  proposalTopic = alphaDist.sampleRandom(gen)
                } else {
                  proposalTopic = resampleTable(gen, docDist, topic, docTopics(topic))
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
              j += 1
            }
            i += 1
          }
          pos += 1
        }
        thq.add(thid)
      }))
      Await.ready(all, 2.hour)
      es.shutdown()

      (pid, ep.withoutVertexAttributes[TC]())
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
  def tokenSampling(gen: Random,
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
    if (gen.nextDouble() < math.min(1.0, pi)) proposalTopic else currentTopic
  }

  // scalastyle:off
  def tokenTopicProb(totalTopicCounter: BDV[Count],
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

  def wordProb(totalTopicCounter: BDV[Count],
    numTerms: Int,
    beta: Double)(termTopicCounter: TC, topic: Int, isAdjustment: Boolean): Double = {
    (termTopicCounter(topic) + beta) / (totalTopicCounter(topic) + beta * numTerms)
  }

  def docProb(totalTopicCounter: BDV[Count],
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
  def wDense(wd: AliasTable[Double],
    topicCounters: BDV[Count],
    numTopics: Int,
    beta: Double,
    betaSum: Double): Unit = topicCounters.synchronized {
    val probs = new Array[Double](numTopics)
    var i = 0
    while (i < numTopics) {
      probs(i) = beta / (topicCounters(i) + betaSum)
      i += 1
    }
    wd.resetDist(probs, null, numTopics)
  }

  /**
   * \frac{{n}_{kw}}{{n}_{k}+\bar{\beta}}
   */
  def wSparse(ws: AliasTable[Double],
    topicCounters: BDV[Count],
    termTopics: TC,
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

  def dDense(dd: AliasTable[Double],
    topicCounters: BDV[Count],
    numTopics: Int,
    alphaRatio: Double,
    alphaAS: Double): Unit = topicCounters.synchronized {
    val probs = new Array[Double](numTopics)
    var i = 0
    while (i < numTopics) {
      probs(i) = alphaRatio * (topicCounters(i) + alphaAS)
      i += 1
    }
    dd.resetDist(probs, null, numTopics)
  }

  def dSparseCached(updatePred: SoftReference[AliasTable[Count]] => Boolean,
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

abstract class LDADocByDoc extends LDAAlgorithm {
  override def updateVertexCounters(sampledCorpus: Graph[TC, TA],
    numTopics: Int,
    inferenceOnly: Boolean = false): GraphImpl[TC, TA] = {
    val dscp = numTopics >> 3
    val graph = sampledCorpus.asInstanceOf[GraphImpl[TC, TA]]
    val vertices = graph.vertices
    val edges = graph.edges
    val conf = edges.context.getConf
    val numThreads = conf.getInt(cs_numThreads, 1)
    val shippedCounters = edges.partitionsRDD.mapPartitions(_.flatMap(Function.tupled((_, ep) => {
      val totalSize = ep.size
      val lcSrcIds = ep.localSrcIds
      val lcDstIds = ep.localDstIds
      val l2g = ep.local2global
      val vattrs = ep.vertexAttrs
      val data = ep.data
      val vertSize = vattrs.length
      val results = new Array[(VertexId, TC)](vertSize)
      val marks = new AtomicIntegerArray(vertSize)

      implicit val es = ExecutionContext.fromExecutorService(Executors.newFixedThreadPool(numThreads))
      val all = Future.traverse(ep.index.iterator)(Function.tupled((_, offset) => Future {
        val si = lcSrcIds(offset)
        val docTuple = (l2g(si), BSV.zeros[Count](numTopics))
        results(si) = docTuple
        val docTopics = docTuple._2
        var pos = offset
        while (pos < totalSize && lcSrcIds(pos) == si) {
          val di = lcDstIds(pos)
          var termTuple = results(di)
          if (termTuple == null && !inferenceOnly) {
            if (marks.getAndDecrement(di) == 0) {
              termTuple = (l2g(di), BSV.zeros[Count](numTopics))
              results(di) = termTuple
              marks.set(di, Int.MaxValue)
            } else {
              while (marks.get(di) <= 0) {}
              termTuple = results(di)
            }
          }
          var termTopics = if (!inferenceOnly) termTuple._2 else null
          val topics = data(pos)
          var i = 0
          while (i < topics.length) {
            val topic = topics(i)
            docTopics(topic) += 1
            if (!inferenceOnly) {
              while (marks.getAndSet(di, -1) < 0) {}
              termTopics match {
                case v: BDV[Count] => v(topic) += 1
                case v: BSV[Count] =>
                  v(topic) += 1
                  if (v.activeSize >= dscp) {
                    termTuple = (l2g(si), toBDV(v))
                    results(di) = termTuple
                    termTopics = termTuple._2
                  }
              }
              marks.set(di, Int.MaxValue)
            }
            i += 1
          }
          pos += 1
        }
      }))
      Await.ready(all, 1.hour)
      es.shutdown()

      results.iterator.filter(_ != null)
    }))).partitionBy(vertices.partitioner.get)

    val partRDD = vertices.partitionsRDD.zipPartitions(shippedCounters, preservesPartitioning=true)(
      (svpIter, cntsIter) => svpIter.map(svp => {
        val results = svp.values
        val index = svp.index
        val marks = new AtomicIntegerArray(results.length)
        implicit val es = ExecutionContext.fromExecutorService(Executors.newFixedThreadPool(numThreads))
        val all = cntsIter.grouped(numThreads * 5).map(batch => Future {
          batch.foreach(Function.tupled((vid, counter) => {
            val i = index.getPos(vid)
            if (marks.getAndDecrement(i) == 0) {
              results(i) = counter
            } else {
              while (marks.getAndSet(i, -1) <= 0) {}
              val agg = results(i)
              results(i) = if (isTermId(vid)) agg match {
                case u: BDV[Count] => counter match {
                  case v: BDV[Count] => u :+= v
                  case v: BSV[Count] => u :+= v
                }
                case u: BSV[Count] => counter match {
                  case v: BDV[Count] => v :+= u
                  case v: BSV[Count] =>
                    u :+= v
                    if (u.activeSize >= dscp) toBDV(u) else u
                }
              } else agg match {
                case u: BSV[Count] => counter match {
                  case v: BSV[Count] => u :+= v
                }
              }
            }
            marks.set(i, Int.MaxValue)
          }))
        })
        Await.ready(Future.sequence(all), 1.hour)
        es.shutdown()
        svp.withValues(results)
      })
    )
    GraphImpl.fromExistingRDDs(vertices.withPartitionsRDD(partRDD), edges)
  }
}

class SparseLDA extends LDADocByDoc {
  override def sampleGraph(corpus: Graph[TC, TA],
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
    val numPartitions = edges.partitions.length
    val partRDD = edges.partitionsRDD.mapPartitions(_.map(Function.tupled((pid, ep) => {
      val alphaRatio = alpha * numTopics / (numTokens + alphaAS * numTopics)
      val betaSum = beta * numTerms
      val totalSize = ep.size
      val lcSrcIds = ep.localSrcIds
      val lcDstIds = ep.localDstIds
      val vattrs = ep.vertexAttrs
      val data = ep.data
      val thq = new ConcurrentLinkedQueue(0 until numThreads)
      val gens = new Array[XORShiftRandom](numThreads)
      val docDists = new Array[FlatDist[Double]](numThreads)
      val mainDists = new Array[FlatDist[Double]](numThreads)
      val alphaK_denoms = calcGlobalCache(topicCounters, alphaAS, betaSum, alphaRatio, numTopics)
      val global = new FlatDist[Double](numTopics, isSparse=false)
      tDense(global, alphaK_denoms, beta, numTopics)

      implicit val es = ExecutionContext.fromExecutorService(Executors.newFixedThreadPool(numThreads))
      val all = Future.traverse(ep.index.iterator)(Function.tupled((_, offset) => Future {
        val thid = thq.poll()
        var gen = gens(thid)
        if (gen == null) {
          gen = new XORShiftRandom(((seed + sampIter) * numPartitions + pid) * numThreads + thid)
          gens(thid) = gen
          docDists(thid) = new FlatDist[Double](numTopics, isSparse=true)
          mainDists(thid) = new FlatDist[Double](numTopics, isSparse=true)
        }
        val docDist = docDists(thid)
        val si = lcSrcIds(offset)
        val docTopics = vattrs(si).asInstanceOf[BSV[Count]]
        val nkd_denoms = calcDocCache(topicCounters, docTopics, betaSum, numTopics)
        dSparse(docDist, nkd_denoms, beta)
        val docAlphaK_Denoms = calcDocAlphaKDenoms(alphaK_denoms, nkd_denoms)
        val mainDist = mainDists(thid)
        var pos = offset
        while (pos < totalSize && lcSrcIds(pos) == si) {
          val di = lcDstIds(pos)
          val termTopics = vattrs(di)
          wSparse(mainDist, termTopics, docAlphaK_Denoms, numTopics)
          val topics = data(pos)
          var i = 0
          while (i < topics.length) {
            topics(i) = tokenSampling(gen, global, docDist, mainDist)
            i += 1
          }
          pos += 1
        }
        thq.add(thid)
      }))
      Await.ready(all, 2.hour)
      es.shutdown()

      (pid, ep.withoutVertexAttributes[TC]())
    })), preservesPartitioning = true)
    GraphImpl.fromExistingRDDs(vertices, edges.withPartitionsRDD(partRDD))
  }

  def tokenSampling(gen: Random,
    t: FlatDist[Double],
    d: FlatDist[Double],
    w: FlatDist[Double]): Int = {
    val wSum = w.norm
    val wdSum = wSum + d.norm
    val distSum = wdSum + t.norm
    val genSum = gen.nextDouble() * distSum
    if (genSum < wSum) {
      w.sampleFrom(genSum, gen)
    } else if (genSum < wdSum) {
      d.sampleFrom(genSum - wSum, gen)
    } else {
      t.sampleFrom(genSum - wdSum, gen)
    }
  }

  def tDense(t: FlatDist[Double],
    alphaK_denoms: BDV[Double],
    beta: Double,
    numTopics: Int): FlatDist[Double] = {
    val probs = new Array[Double](numTopics)
    var i = 0
    while (i < numTopics) {
      probs(i) = beta * alphaK_denoms(i)
      i += 1
    }
    t.resetDist(probs, null, numTopics)
  }

  def dSparse(d: FlatDist[Double],
    nkd_denoms: BSV[Double],
    beta: Double): FlatDist[Double] = {
    val used = nkd_denoms.used
    val index = nkd_denoms.index
    val data = nkd_denoms.data
    val probs = new Array[Double](used)
    var i = 0
    while (i < used) {
      probs(i) = beta * nkd_denoms(i)
      i += 1
    }
    d.resetDist(probs, index, used)
  }

  def wSparse(w: FlatDist[Double],
    termTopics: TC,
    docAlphaK_Denoms: BDV[Double],
    numTopics: Int): FlatDist[Double] = termTopics match {
    case v: BDV[Count] =>
      val probs = new Array[Double](numTopics)
      val space = new Array[Int](numTopics)
      var psize = 0
      var i = 0
      while (i < numTopics) {
        val cnt = v(i)
        if (cnt > 0) {
          probs(psize) = docAlphaK_Denoms(i) * cnt
          space(psize) = i
          psize += 1
        }
        i += 1
      }
      w.resetDist(probs, space, psize)
    case v: BSV[Count] =>
      val used = v.used
      val index = v.index
      val data = v.data
      val probs = new Array[Double](used)
      var i = 0
      while (i < used) {
        probs(i) = docAlphaK_Denoms(index(i)) * data(i)
        i += 1
      }
      w.resetDist(probs, index, used)
  }

  def calcGlobalCache(topicCounters: BDV[Count],
    alphaAS: Double,
    betaSum: Double,
    alphaRatio: Double,
    numTopics: Int): BDV[Double] = {
    val arr = new Array[Double](numTopics)
    var i = 0
    while (i < numTopics) {
      val nt = topicCounters(i)
      arr(i) = alphaRatio * (nt + alphaAS) / (nt + betaSum)
      i += 1
    }
    new BDV(arr)
  }

  def calcDocCache(topicCounters: BDV[Count],
    docTopics: BSV[Count],
    betaSum: Double,
    numTopics: Int): BSV[Double] = {
    val used = docTopics.used
    val index = docTopics.index
    val data = docTopics.data
    val arr = new Array[Double](used)
    var i = 0
    while (i < used) {
      arr(i) = data(i) / (topicCounters(i) + betaSum)
      i += 1
    }
    new BSV(index, arr, used, numTopics)
  }

  def calcDocAlphaKDenoms(alphaK_denoms: BDV[Double],
    nkd_denoms: BSV[Double]): BDV[Double] = {
    val bdv = alphaK_denoms.copy
    val used = nkd_denoms.used
    val index = nkd_denoms.index
    val data = nkd_denoms.data
    var i = 0
    while (i < used) {
      bdv(index(i)) += data(i)
      i += 1
    }
    bdv
  }
}
