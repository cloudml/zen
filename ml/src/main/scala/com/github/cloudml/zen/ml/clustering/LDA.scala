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

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, sum => brzSum, Vector => BV}
import com.github.cloudml.zen.ml.DBHPartitioner
import com.github.cloudml.zen.ml.clustering.LDA._
import com.github.cloudml.zen.ml.clustering.LDADefines._
import com.github.cloudml.zen.ml.util.{FTree, DiscreteSampler, XORShiftRandom, AliasTable}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.graphx._
import org.apache.spark.mllib.linalg.{SparseVector => SSV, Vector => SV}
import org.apache.spark.mllib.linalg.distributed.{MatrixEntry, RowMatrix}
import org.apache.spark.rdd.RDD
import org.apache.spark.serializer.KryoRegistrator
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.collection.AppendOnlyMap
import org.apache.spark.Logging


abstract class LDA(
  @transient private var corpus: Graph[VD, ED],
  private val numTopics: Int,
  private val numTerms: Int,
  private val numDocs: Long,
  private val numTokens: Long,
  private var alpha: Double,
  private var beta: Double,
  private var alphaAS: Double,
  private var storageLevel: StorageLevel) extends Serializable with Logging {

  @transient private var seed = new XORShiftRandom().nextInt()
  @transient private var totalTopicCounter = collectTopicCounter()

  def setAlpha(alpha: Double): this.type = {
    this.alpha = alpha
    this
  }

  def setBeta(beta: Double): this.type = {
    this.beta = beta
    this
  }

  def setAlphaAS(alphaAS: Double): this.type = {
    this.alphaAS = alphaAS
    this
  }

  def setStorageLevel(storageLevel: StorageLevel): this.type = {
    this.storageLevel = storageLevel
    this
  }

  def setSeed(newSeed: Int): this.type = {
    this.seed = newSeed
    this
  }

  def getCorpus: Graph[VD, ED] = corpus

  def termVertices: VertexRDD[VD] = corpus.vertices.filter(t => !isDocId(t._1))

  def docVertices: VertexRDD[VD] = corpus.vertices.filter(t => isDocId(t._1))

  private def scConf = corpus.edges.context.getConf

  private def collectTopicCounter(): BDV[Count] = {
    val gtc = termVertices.map(_._2).aggregate(BDV.zeros[Count](numTopics))(_ :+= _, _ :+= _)
    val count = gtc.activeValuesIterator.map(_.toLong).sum
    assert(count == numTokens)
    gtc
  }

  def runGibbsSampling(totalIter: Int): Unit = {
    val sc = corpus.edges.context
    val pplx = scConf.getBoolean(cs_calcPerplexity, false)
    val saveIntv = scConf.getInt(cs_saveInterval, 0)
    if (pplx) {
      println(s"Before Gibbs sampling: perplexity=${perplexity()}")
    }
    for (iter <- 1 to totalIter) {
      println(s"Start Gibbs sampling (Iteration $iter/$totalIter)")
      val startedAt = System.nanoTime()
      gibbsSampling(iter)
      val elapsedSeconds = (System.nanoTime() - startedAt) / 1e9
      if (pplx) {
        println(s"Gibbs sampling (Iteration $iter/$totalIter): perplexity=${perplexity()}")
      }
      println(s"End Gibbs sampling (Iteration $iter/$totalIter) takes: $elapsedSeconds secs")
      if (saveIntv > 0 && iter % saveIntv == 0) {
        val outputPath = scConf.get(cs_outputpath)
        saveModel().save(sc, s"$outputPath-iter$iter", isTransposed=true)
        println(s"Saved model after iter-$iter")
      }
    }
  }

  private def gibbsSampling(sampIter: Int): Unit = {
    val sc = corpus.edges.context
    val chkptIntv = scConf.getInt(cs_chkptInterval, 0)
    val prevCorpus = corpus
    val sampledCorpus = sampleTokens(corpus, totalTopicCounter, sampIter + seed,
      numTokens, numTopics, numTerms, alpha, alphaAS, beta)
    sampledCorpus.edges.persist(storageLevel).setName(s"edges-$sampIter").count()
    prevCorpus.edges.unpersist(blocking=false)
    corpus = updateCounter(sampledCorpus, numTopics)
    corpus.vertices.persist(storageLevel).setName(s"vertices-$sampIter")
    if (chkptIntv > 0 && sampIter % chkptIntv == 1 && sc.getCheckpointDir.isDefined) {
      corpus.checkpoint()
      corpus.edges.first()
    }
    totalTopicCounter = collectTopicCounter()
    prevCorpus.vertices.unpersist(blocking=false)
  }

  protected def sampleTokens(corpus: Graph[VD, ED],
    totalTopicCounter: BDV[Count],
    pseudoIter: Int,
    numTokens: Long,
    numTopics: Int,
    numTerms: Int,
    alpha: Double,
    alphaAS: Double,
    beta: Double): Graph[VD, ED]

  /**
   * Save the term-topic related model
   * @param runIter saved more these iters' averaged model
   */
  def saveModel(runIter: Int = 0): DistributedLDAModel = {
    var ttcSum: RDD[(VertexId, VD)] = termVertices
    ttcSum.persist(storageLevel).count()
    for (iter <- 1 to runIter) {
      println(s"Save TopicModel (Iteration $iter/$runIter)")
      gibbsSampling(-iter)
      val newTtcSum = ttcSum.join(termVertices).map {
        case (term, (a, b)) => (term, a += b)
      }
      newTtcSum.persist(storageLevel).count()
      ttcSum.unpersist(blocking=false)
      ttcSum = newTtcSum
    }
    val ttc = if (runIter == 0) {
      ttcSum
    } else {
      val rand = new XORShiftRandom()
      val aver = ttcSum.mapValues(_.mapValues(s => {
        val mid = s.toDouble / (runIter + 1)
        val l = math.floor(mid)
        if (rand.nextDouble() > mid - l) l else l + 1
      }.toInt))
      aver.persist(storageLevel).count()
      ttcSum.unpersist(blocking=false)
      aver
    }
    val gtc = ttc.map(_._2).aggregate(BDV.zeros[Count](numTopics))(_ :+= _, _ :+= _)
    new DistributedLDAModel(gtc, ttc, numTopics, numTerms, alpha, beta, alphaAS)
  }

  def mergeDuplicateTopic(threshold: Double = 0.95D): Map[Int, Int] = {
    val rows = termVertices.map(t => t._2).map { bsv =>
      val length = bsv.length
      val used = bsv.activeSize
      val index = bsv.index.slice(0, used)
      val data = bsv.data.slice(0, used).map(_.toDouble)
      new SSV(length, index, data).asInstanceOf[SV]
    }
    val simMatrix = new RowMatrix(rows).columnSimilarities()
    val minMap = simMatrix.entries.filter { case MatrixEntry(row, column, sim) =>
      sim > threshold && row != column
    }.map { case MatrixEntry(row, column, sim) =>
      (column.toInt, row.toInt)
    }.groupByKey().map { case (topic, simTopics) =>
      (topic, simTopics.min)
    }.collect().toMap
    if (minMap.nonEmpty) {
      val mergingCorpus = corpus.mapEdges{
        _.attr.map(topic => minMap.getOrElse(topic, topic))
      }
      corpus = updateCounter(mergingCorpus, numTopics)
    }
    minMap
  }

  /**
   * the multiplcation between word distribution among all topics and the corresponding doc
   * distribution among all topics:
   * p(w)=\sum_{k}{p(k|d)*p(w|k)}=
   * \sum_{k}{\frac{{n}_{kw}+{\beta }_{w}} {{n}_{k}+\bar{\beta }} \frac{{n}_{kd}+{\alpha }_{k}}{\sum{{n}_{k}}+
   * \bar{\alpha }}}=
   * \sum_{k} \frac{{\alpha }_{k}{\beta }_{w}  + {n}_{kw}{\alpha }_{k} + {n}_{kd}{\beta }_{w} + {n}_{kw}{n}_{kd}}
   * {{n}_{k}+\bar{\beta }} \frac{1}{\sum{{n}_{k}}+\bar{\alpha }}}
   * \exp^{-(\sum{\log(p(w))})/N}
   * N is the number of tokens in corpus
   */
  def perplexity(): Double = {
    val totalTopicCounter = this.totalTopicCounter
    val alpha = this.alpha
    val beta = this.beta
    val alpha_bar = this.numTopics * alpha
    val beta_bar = this.numTerms * beta
    val numTokens = this.numTokens

    // \frac{{\alpha }_{k}{\beta }_{w}}{{n}_{k}+\bar{\beta }}
    val tDenseSum = totalTopicCounter.valuesIterator.map(c => alpha * beta / (c + beta_bar)).sum

    val termProb = corpus.mapVertices { (vid, counter) =>
      val probDist = if (isDocId(vid)) {
        counter.mapActivePairs((t, c) => c * beta / (totalTopicCounter(t) + beta_bar))
      } else {
        counter.mapActivePairs((t, c) => c * alpha / (totalTopicCounter(t) + beta_bar))
      }
      (counter, brzSum(probDist), brzSum(counter))
    }.mapTriplets { triplet =>
      val (termTopicCounter, wSparseSum, _) = triplet.srcAttr
      val (docTopicCounter, dSparseSum, docSize) = triplet.dstAttr
      val dwCooccur = triplet.attr.length

      // \frac{{n}_{kw}{n}_{kd}}{{n}_{k}+\bar{\beta}}
      val dwSparseSum = docTopicCounter.activeIterator.map { case (t, c) =>
        c * termTopicCounter(t) / (totalTopicCounter(t) + beta_bar)
      }.sum
      val prob = (tDenseSum + wSparseSum + dSparseSum + dwSparseSum) / (docSize + alpha_bar)

      dwCooccur * Math.log(prob)
    }.edges.map(_.attr).sum()

    math.exp(-1 * termProb / numTokens)
  }

}

object LDA {
  /**
   * LDA training
   * @param docs       RDD of documents, which are term (word) count vectors paired with IDs.
   *                   The term count vectors are "bags of words" with a fixed-size vocabulary
   *                   (where the vocabulary size is the length of the vector).
   *                   Document IDs must be unique and >= 0.
   * @param totalIter  the number of iterations
   * @param numTopics  the number of topics (5000+ for large data)
   * @param alpha      recommend to be (5.0 /numTopics)
   * @param beta       recommend to be in range 0.001 - 0.1
   * @param alphaAS    recommend to be in range 0.01 - 1.0
   * @param storageLevel StorageLevel that the LDA Model RDD uses
   * @return DistributedLDAModel
   */
  def train(
    docs: RDD[BOW],
    totalIter: Int,
    numTopics: Int,
    alpha: Double,
    beta: Double,
    alphaAS: Double,
    storageLevel: StorageLevel): DistributedLDAModel = {
    val conf = docs.context.getConf
    val LDAAlgorithm = conf.get(cs_LDAAlgorithm, "fastlda")
    val lda: LDA = LDAAlgorithm match {
      case "lightlda" =>
        println("using LightLDA sampling algorithm.")
        LightLDA(docs, numTopics, alpha, beta, alphaAS, storageLevel)
      case "fastlda" =>
        println("using FastLDA sampling algorithm.")
        FastLDA(docs, numTopics, alpha, beta, alphaAS, storageLevel)
      case _ =>
        throw new NoSuchMethodException("No this algorithm or not implemented.")
    }
    lda.runGibbsSampling(totalIter)
    lda.saveModel()
  }

  def incrementalTrain(
    docs: RDD[BOW],
    computedModel: LocalLDAModel,
    totalIter: Int,
    storageLevel: StorageLevel): DistributedLDAModel = {
    val conf = docs.context.getConf
    val LDAAlgorithm = conf.get(cs_LDAAlgorithm, "fastlda")
    val numTopics = computedModel.numTopics
    val alpha = computedModel.alpha
    val beta = computedModel.beta
    val alphaAS = computedModel.alphaAS
    val broadcastModel = docs.context.broadcast(computedModel)
    val lda: LDA = LDAAlgorithm match {
      case "lightlda" =>
        println("using LightLDA sampling algorithm.")
        LightLDA(docs, numTopics, alpha, beta, alphaAS, storageLevel, broadcastModel)
      case "fastlda" =>
        println("using FastLDA sampling algorithm.")
        FastLDA(docs, numTopics, alpha, beta, alphaAS, storageLevel, broadcastModel)
      case _ =>
        throw new NoSuchMethodException("No this algorithm or not implemented.")
    }
    broadcastModel.unpersist(blocking=false)
    lda.runGibbsSampling(totalIter)
    lda.saveModel()
  }

  private[ml] def initializeCorpus(
    docs: RDD[BOW],
    numTopics: Int,
    storageLevel: StorageLevel,
    computedModel: Broadcast[LocalLDAModel] = null): Graph[VD, ED] = {
    val conf = docs.context.getConf
    val partStrategy = conf.get(cs_partStrategy, "dbh")
    val edges = docs.mapPartitionsWithIndex((pid, iter) => {
      val gen = new XORShiftRandom(pid + 117)
      iter.flatMap { case (docId, doc) =>
        if (computedModel == null) {
          initializeEdges(gen, doc, docId, numTopics)
        } else {
          initEdgesWithComputedModel(gen, doc, docId, numTopics, computedModel.value)
        }
      }
    })
    var initCorpus: Graph[VD, ED] = Graph.fromEdges(edges, null, storageLevel, storageLevel)
    val numEdges = initCorpus.edges.persist(storageLevel).setName("initEdges").count()
    println(s"edges in the corpus: $numEdges")
    docs.unpersist(blocking=false)
    initCorpus = partStrategy match {
      case "dbh" =>
        println("using Degree-based Hashing partition strategy.")
        DBHPartitioner.partitionByDBH[VD, ED](initCorpus, storageLevel)
      case "edge2d" =>
        println("using Edge2D partition strategy.")
        initCorpus.partitionBy(PartitionStrategy.EdgePartition2D)
      case _ =>
        throw new NoSuchMethodException("No this algorithm or not implemented.")
    }
    val corpus = updateCounter(initCorpus, numTopics)
    corpus.vertices.persist(storageLevel).setName("initVertices")
    corpus
  }

  private def initializeEdges(
    gen: Random,
    doc: BSV[Int],
    docId: DocId,
    numTopics: Int): Iterator[Edge[ED]] = {
    val newDocId: DocId = genNewDocId(docId)
    doc.activeIterator.filter(_._2 > 0).map { case (termId, counter) =>
      val topics = new Array[Int](counter)
      for (i <- 0 until counter) {
        topics(i) = gen.nextInt(numTopics)
      }
      Edge(termId, newDocId, topics)
    }
  }

  private def initEdgesWithComputedModel(
    gen: Random,
    doc: BSV[Int],
    docId: DocId,
    numTopics: Int,
    computedModel: LocalLDAModel): Iterator[Edge[ED]] = {
    val newDocId: DocId = genNewDocId(docId)
    computedModel.setSeed(gen.nextInt())
    val tokens = computedModel.vector2Array(doc)
    val topics = new Array[Int](tokens.length)
    var docTopicCounter = computedModel.uniformDistSampler(tokens, topics)
    for (t <- 0 until 15) {
      docTopicCounter = computedModel.sampleTokens(docTopicCounter, tokens, topics)
    }
    doc.activeIterator.filter(_._2 > 0).map { case (term, counter) =>
      val ev = topics.zipWithIndex.filter { case (topic, offset) =>
        term == tokens(offset)
      }.map(_._1)
      Edge(term, newDocId, ev)
    }
  }

  // make docId always be negative, so that the doc vertex always be the dest vertex
  @inline private def genNewDocId(docId: Long): Long = {
    assert(docId >= 0)
    -(docId + 1L)
  }

  @inline private def isDocId(id: Long): Boolean = id < 0L

  private[ml] def sampleSV(
    gen: Random,
    table: AliasTable,
    sv: VD,
    currentTopic: Int,
    currentTopicCounter: Int = 0,
    numSampling: Int = 0): Int = {
    val docTopic = table.sample(gen)
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

  private def updateCounter(corpus: Graph[VD, ED],
    numTopics: Int): Graph[VD, ED] = {
    val newCounter = corpus.edges.mapPartitions(iter =>
      iter.flatMap(edge => {
        val vid = edge.srcId
        val did = edge.dstId
        val topics = edge.attr
        Iterator((vid, topics), (did, topics))
      })
    ).aggregateByKey(BSV.zeros[Count](numTopics), corpus.vertices.partitioner.get)((agg, cur) => {
      for (t <- cur) {
        agg(t) += 1
      }
      agg
    }, _ += _)
    corpus.joinVertices(newCounter)((_, _, counter) => counter)
  }
}

private[ml] class LDAKryoRegistrator extends KryoRegistrator {
  def registerClasses(kryo: com.esotericsoftware.kryo.Kryo) {
    kryo.register(classOf[BSV[Count]])
    kryo.register(classOf[BSV[Double]])

    kryo.register(classOf[BDV[Count]])
    kryo.register(classOf[BDV[Double]])

    kryo.register(classOf[ED])
    kryo.register(classOf[VD])
    kryo.register(classOf[BOW])

    kryo.register(classOf[Random])
    kryo.register(classOf[LDA])
    kryo.register(classOf[LocalLDAModel])
  }
}

class FastLDA(
  corpus: Graph[VD, ED],
  numTopics: Int,
  numTerms: Int,
  numDocs: Long,
  numTokens: Long,
  alpha: Double,
  beta: Double,
  alphaAS: Double,
  storageLevel: StorageLevel)
  extends LDA(corpus, numTopics, numTerms, numDocs, numTokens, alpha, beta, alphaAS, storageLevel) {

  override protected def sampleTokens(corpus: Graph[VD, ED],
    totalTopicCounter: BDV[Count],
    pseudoIter: Int,
    numTokens: Long,
    numTopics: Int,
    numTerms: Int,
    alpha: Double,
    alphaAS: Double,
    beta: Double): Graph[VD, ED] = {
    val sampl = corpus.edges.context.getConf.get(cs_accelMethod, "alias")
    val numPartitions = corpus.edges.partitions.length
    corpus.mapTriplets((pid, iter) => {
      val gen = new XORShiftRandom(numPartitions * pseudoIter + pid)
      // table/ftree is a per term data structure
      // in GraphX, edges in a partition are clustered by source IDs (term id in this case)
      // so, use below simple cache to avoid calculating table each time
      val lastSampler: DiscreteSampler = sampl match {
        case "alias" => new AliasTable(numTopics)
        case "ftree" | "hybrid" => new FTree(numTopics, isSparse=true)
      }
      var lastVid: VertexId = -1L
      val globalSampler: DiscreteSampler = sampl match {
        case "ftree" => new FTree(numTopics, isSparse=false)
        case "alias" | "hybrid" => new AliasTable(numTopics)
      }
      val tdt = tDense(totalTopicCounter, numTokens, numTerms, alpha, alphaAS, beta)
      globalSampler.resetDist(tdt._2, tdt._1)
      val docCdf = new Array[Double](numTopics)
      iter.map(triplet => {
        val termId = triplet.srcId
        // val docId = triplet.dstId
        val termTopicCounter = triplet.srcAttr
        val docTopicCounter = triplet.dstAttr
        val topics = triplet.attr
        for (i <- topics.indices) {
          val currentTopic = topics(i)
          dSparse(totalTopicCounter, termTopicCounter, docTopicCounter, docCdf,
            currentTopic, numTokens, numTerms, alpha, alphaAS, beta)
          if (lastVid != termId) {
            lastVid = termId
            val wst = wSparse(totalTopicCounter, termTopicCounter, numTokens,
              numTerms, alpha, alphaAS, beta)
            lastSampler.resetDist(wst._2, wst._1)
          }
          def newAlpha(t: Int) = alpha * numTopics * (totalTopicCounter(t) + alphaAS) /
            (numTokens + alphaAS * numTopics)
          def deltaDenom(t: Int) = {
            val denom = totalTopicCounter(t) + numTerms * beta
            denom * (denom - 1)
          }
          def wNumer(t: Int) = termTopicCounter(t) - totalTopicCounter(t) - beta * numTerms
          globalSampler.update(currentTopic, newAlpha(currentTopic) * beta / deltaDenom(currentTopic))
          lastSampler.update(currentTopic, newAlpha(currentTopic) * wNumer(currentTopic) / deltaDenom(currentTopic))
          val newTopic = tokenSampling(gen, globalSampler, lastSampler, docCdf, termTopicCounter,
            docTopicCounter, currentTopic)
          globalSampler.update(newTopic, newAlpha(newTopic) * beta / deltaDenom(newTopic))
          lastSampler.update(newTopic, newAlpha(newTopic) * wNumer(newTopic) / deltaDenom(newTopic))
          if (newTopic != currentTopic) {
            topics(i) = newTopic
          }
        }
        topics
      })
    }, TripletFields.All)
  }

  private def tokenSampling(gen: Random,
    t: DiscreteSampler,
    w: DiscreteSampler,
    dData: Array[Double],
    termTopicCounter: VD,
    docTopicCounter: VD,
    currentTopic: Int): Int = {
    val tSum = t.norm
    val wSum = w.norm
    val dSum = dData(docTopicCounter.used - 1)
    val distSum = tSum + wSum + dSum
    val genSum = gen.nextDouble() * distSum
    if (genSum < dSum) {
      val dGenSum = gen.nextDouble() * dSum
      val index = docTopicCounter.index
      val used = docTopicCounter.used
      val pos = binarySearchInterval(dData, dGenSum, 0, used, greater=true)
      index(pos)
    } else if (genSum < (dSum + wSum)) {
      w match {
        case wt: AliasTable => sampleSV(gen, wt, termTopicCounter, currentTopic)
        case wf: FTree => wf.sample(gen)
      }
    } else {
      t.sample(gen)
    }
  }

  /**
   * dense part in the decomposed sampling formula:
   * t = \frac{{\beta }_{w} \bar{\alpha} ( {n}_{k}^{-di} + \acute{\alpha} ) } {({n}_{k}^{-di}+\bar{\beta})
   * ({\sum{n}_{k}^{-di} +\bar{\acute{\alpha}}})}
   */
  private def tDense(totalTopicCounter: BDV[Count],
    numTokens: Long,
    numTerms: Int,
    alpha: Double,
    alphaAS: Double,
    beta: Double): (Double, BDV[Double]) = {
    val numTopics = totalTopicCounter.length
    val t = BDV.zeros[Double](numTopics)
    val alphaSum = alpha * numTopics
    val termSum = numTokens - 1 + alphaAS * numTopics
    val betaSum = numTerms * beta
    var sum = 0D
    for (topic <- 0 until numTopics) {
      val last = beta * alphaSum * (totalTopicCounter(topic) + alphaAS) /
        ((totalTopicCounter(topic) + betaSum) * termSum)
      t(topic) = last
      sum += last
    }
    (sum, t)
  }

  /**
   * word related sparse part in the decomposed sampling formula:
   * w = \frac{ {n}_{kw}^{-di} \bar{\alpha} ( {n}_{k}^{-di} + \acute{\alpha} )}{({n}_{k}^{-di}+\bar{\beta})
   * ({\sum{n}_{k}^{-di} +\bar{\acute{\alpha}}})}
   */
  private def wSparse(totalTopicCounter: BDV[Count],
    termTopicCounter: VD,
    numTokens: Long,
    numTerms: Int,
    alpha: Double,
    alphaAS: Double,
    beta: Double): (Double, BSV[Double]) = {
    val numTopics = totalTopicCounter.length
    val alphaSum = alpha * numTopics
    val termSum = numTokens - 1D + alphaAS * numTopics
    val betaSum = numTerms * beta
    val w = BSV.zeros[Double](numTopics)
    var sum = 0D
    termTopicCounter.activeIterator.filter(_._2 > 0).foreach { t =>
      val topic = t._1
      val count = t._2
      val last = count * alphaSum * (totalTopicCounter(topic) + alphaAS) /
        ((totalTopicCounter(topic) + betaSum) * termSum)
      w(topic) = last
      sum += last
    }
    (sum, w)
  }

  /**
   * doc related sparse part in the decomposed sampling formula:
   * d =  \frac{{n}_{kd} ^{-di}({\sum{n}_{k}^{-di} + \bar{\acute{\alpha}}})({n}_{kw}^{-di}+{\beta}_{w})}
   * {({n}_{k}^{-di}+\bar{\beta})({\sum{n}_{k}^{-di} +\bar{\acute{\alpha}}})}
   * =  \frac{{n}_{kd} ^{-di}({n}_{kw}^{-di}+{\beta}_{w})}{({n}_{k}^{-di}+\bar{\beta}) }
   */
  private def dSparse(totalTopicCounter: BDV[Count],
    termTopicCounter: VD,
    docTopicCounter: VD,
    d: Array[Double],
    currentTopic: Int,
    numTokens: Long,
    numTerms: Int,
    alpha: Double,
    alphaAS: Double,
    beta: Double): Unit = {
    val index = docTopicCounter.index
    val data = docTopicCounter.data
    val used = docTopicCounter.used

    // val termSum = numTokens - 1D + alphaAS * numTopics
    val betaSum = numTerms * beta
    var sum = 0D
    for (i <- 0 until used) {
      val topic = index(i)
      val count = data(i)
      val adjustment = if (currentTopic == topic) -1 else 0
      val last = (count + adjustment) * (termTopicCounter(topic) + adjustment + beta) /
        (totalTopicCounter(topic) + adjustment + betaSum)
      // val lastD = (count + adjustment) * termSum * (termTopicCounter(topic) + adjustment + beta) /
      //  ((totalTopicCounter(topic) + adjustment + betaSum) * termSum)
      sum += last
      d(i) = sum
    }
  }
}

class LightLDA(
  corpus: Graph[VD, ED],
  numTopics: Int,
  numTerms: Int,
  numDocs: Long,
  numTokens: Long,
  alpha: Double,
  beta: Double,
  alphaAS: Double,
  storageLevel: StorageLevel)
  extends LDA(corpus, numTopics, numTerms, numDocs, numTokens, alpha, beta, alphaAS, storageLevel) {

  override protected def sampleTokens(corpus: Graph[VD, ED],
    totalTopicCounter: BDV[Count],
    pseudoIter: Int,
    numTokens: Long,
    numTopics: Int,
    numTerms: Int,
    alpha: Double,
    alphaAS: Double,
    beta: Double): Graph[VD, ED] = {
    val numPartitions = corpus.edges.partitions.length
    corpus.mapTriplets((pid, iter) => {
      val gen = new XORShiftRandom(numPartitions * pseudoIter + pid)
      val docTableCache = new AppendOnlyMap[VertexId, SoftReference[(Double, AliasTable)]]()

      // table is a per term data structure
      // in GraphX, edges in a partition are clustered by source IDs (term id in this case)
      // so, use below simple cache to avoid calculating table each time
      val lastTable = new AliasTable(numTopics.toInt)
      var lastVid: VertexId = -1L
      var lastWSum = 0D

      val p = tokenTopicProb(totalTopicCounter, beta, alpha,
        alphaAS, numTokens, numTerms) _
      val dPFun = docProb(totalTopicCounter, alpha, alphaAS, numTokens) _
      val wPFun = wordProb(totalTopicCounter, numTerms, beta) _

      var dD: AliasTable = null
      var dDSum = 0D
      var wD: AliasTable = null
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
          dD = AliasTable.generateAlias(dv._2, dDSum)

          dv = wDense(totalTopicCounter, numTerms, beta)
          wDSum = dv._1
          wD = AliasTable.generateAlias(dv._2, wDSum)
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
                proposalTopic = dD.sample(gen)
              }
              else {
                proposalTopic = docTopicCounter.synchronized {
                  sampleSV(gen, d, docTopicCounter, currentTopic)
                }
              }
              dPFun
            } else {
              val table = if (gen.nextDouble() < wSum / (wSum + wDSum)) w else wD
              proposalTopic = table.sample(gen)
              wPFun
            }

            val newTopic = tokenSampling(gen, docTopicCounter, termTopicCounter, docProposal,
              currentTopic, proposalTopic, q, p)
            assert(newTopic >= 0 && newTopic < numTopics)
            if (newTopic != currentTopic) {
              topics(i) = newTopic
            }
          }
        }
        topics
      })
    }, TripletFields.All)
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
  private def tokenTopicProb(totalTopicCounter: BDV[Count],
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

  private def wordProb(totalTopicCounter: BDV[Count],
    numTerms: Int,
    beta: Double)(termTopicCounter: VD, topic: Int, isAdjustment: Boolean): Double = {
    (termTopicCounter(topic) + beta) / (totalTopicCounter(topic) + beta * numTerms)
  }

  private def docProb(totalTopicCounter: BDV[Count],
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
  private def wSparse(totalTopicCounter: BDV[Count],
    termTopicCounter: VD,
    numTerms: Int,
    beta: Double): (Double, BV[Double]) = {
    val numTopics = termTopicCounter.length
    val termSum = beta * numTerms
    val w = BSV.zeros[Double](numTopics)
    var sum = 0D
    termTopicCounter.activeIterator.foreach { t =>
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
  private def wDense(totalTopicCounter: BDV[Count],
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

  private def dSparse(docTopicCounter: VD): (Double, BV[Double]) = {
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

  private def dDense(totalTopicCounter: BDV[Count],
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

  private def docTable(
    updateFunc: SoftReference[(Double, AliasTable)] => Boolean,
    cacheMap: AppendOnlyMap[VertexId, SoftReference[(Double, AliasTable)]],
    docTopicCounter: VD,
    docId: VertexId): (Double, AliasTable) = {
    val cacheD = cacheMap(docId)
    if (!updateFunc(cacheD)) {
      cacheD.get
    } else {
      docTopicCounter.synchronized {
        val sv = dSparse(docTopicCounter)
        val d = (sv._1, AliasTable.generateAlias(sv._2, sv._1))
        cacheMap.update(docId, new SoftReference(d))
        d
      }
    }
  }

  private def wordTable(table: AliasTable,
    totalTopicCounter: BDV[Count],
    termTopicCounter: VD,
    termId: VertexId,
    numTerms: Int,
    beta: Double): Double = {
    val sv = wSparse(totalTopicCounter, termTopicCounter, numTerms, beta)
    AliasTable.generateAlias(sv._2, sv._1, table)
    sv._1
  }
}

object FastLDA {
  def apply(bowDocs: RDD[BOW],
    numTopics: Int,
    alpha: Double,
    beta: Double,
    alphaAS: Double,
    storageLevel: StorageLevel,
    computedModel: Broadcast[LocalLDAModel] = null): FastLDA = {
    val numTerms = bowDocs.first()._2.size
    val numDocs = bowDocs.count()
    val corpus = initializeCorpus(bowDocs, numTopics, storageLevel, computedModel)
    val numTokens = corpus.edges.map(e => e.attr.length.toLong).reduce(_ + _)
    new FastLDA(corpus, numTopics, numTerms, numDocs, numTokens, alpha, beta, alphaAS, storageLevel)
  }
}

object LightLDA {
  def apply(bowDocs: RDD[BOW],
    numTopics: Int,
    alpha: Double,
    beta: Double,
    alphaAS: Double,
    storageLevel: StorageLevel,
    computedModel: Broadcast[LocalLDAModel] = null): LightLDA = {
    val numTerms = bowDocs.first()._2.size
    val numDocs = bowDocs.count()
    val corpus = initializeCorpus(bowDocs, numTopics, storageLevel, computedModel)
    val numTokens = corpus.edges.map(e => e.attr.length.toLong).reduce(_ + _)
    new LightLDA(corpus, numTopics, numTerms, numDocs, numTokens, alpha, beta, alphaAS, storageLevel)
  }
}
