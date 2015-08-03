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
import com.github.cloudml.zen.ml.clustering.LDAUtils._
import com.github.cloudml.zen.ml.util.{XORShiftRandom, AliasTable}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.graphx._
import org.apache.spark.graphx.impl.GraphImpl
import org.apache.spark.mllib.linalg.{SparseVector => SSV, Vector => SV}
import org.apache.spark.mllib.linalg.distributed.{MatrixEntry, RowMatrix}
import org.apache.spark.rdd.RDD
import org.apache.spark.serializer.KryoRegistrator
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.collection.AppendOnlyMap
import org.apache.spark.Logging


abstract class LDA(
  @transient protected var corpus: Graph[VD, ED],
  protected val numTopics: Int,
  protected val numTerms: Int,
  protected val numDocs: Long,
  protected val numTokens: Long,
  protected var alpha: Float,
  protected var beta: Float,
  protected var alphaAS: Float,
  protected var storageLevel: StorageLevel) extends Serializable with Logging {

  @transient protected var seed = new XORShiftRandom().nextInt()
  @transient protected var totalTopicCounter = collectTopicCounter()

  def setAlpha(alpha: Float): this.type = {
    this.alpha = alpha
    this
  }

  def setBeta(beta: Float): this.type = {
    this.beta = beta
    this
  }

  def setAlphaAS(alphaAS: Float): this.type = {
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

  def termVertices: VertexRDD[VD] = corpus.vertices.filter(_._1 >= 0)

  def docVertices: VertexRDD[VD] = corpus.vertices.filter(_._1 < 0)

  private def collectTopicCounter(): BDV[Count] = {
    termVertices.map(_._2).aggregate(BDV.zeros[Count](numTopics))(_ :+= _, _ :+= _)
  }

  def runGibbsSampling(totalIter: Int): Unit = {
    // logInfo(s"Before Gibbs sampling: pplx=${perplexity()}")
    for (iter <- 1 to totalIter) {
      logInfo(s"Start Gibbs sampling (Iteration $iter/$totalIter)")
      val startedAt = System.nanoTime()
      gibbsSampling(iter)
      val elapsedSeconds = (System.nanoTime() - startedAt) / 1e9
      // logInfo(s"Gibbs sampling (Iteration $iter/$totalIter): pplx=${perplexity()}")
      logInfo(s"End Gibbs sampling (Iteration $iter/$totalIter) takes: $elapsedSeconds secs")
    }
  }

  private def gibbsSampling(sampIter: Int): Unit = {
    val prevCorpus = corpus
    val sampledCorpus = sampleTokens(sampIter + seed)
    sampledCorpus.persist(storageLevel)
    sampledCorpus.vertices.setName(s"sampledVertices-$sampIter")
    sampledCorpus.edges.setName(s"sampledEdges-$sampIter")
    corpus = updateCounter(sampledCorpus, numTopics)
    prevCorpus.unpersist(blocking = false)
    if (sampIter % 10 == 0) {
      corpus.checkpoint()
    }
    corpus.persist(storageLevel)
    corpus.vertices.setName(s"vertices-$sampIter")
    corpus.edges.setName(s"edges-$sampIter")
    totalTopicCounter = collectTopicCounter()
    sampledCorpus.unpersist(blocking = false)
  }

  protected def sampleTokens(pseudoIter: Int): Graph[VD, (ED, ED)]

  /**
   * Save the term-topic related model
   * @param saveIter saved these iters' averaged model
   */
  def saveModel(saveIter: Int = 1): DistributedLDAModel = {
    var termTopicCounter: RDD[(VertexId, VD)] = null
    for (iter <- 1 to saveIter) {
      logInfo(s"Save TopicModel (Iteration $iter/$saveIter)")
      var previousTermTopicCounter = termTopicCounter
      gibbsSampling(iter)
      val newTermTopicCounter = termVertices
      termTopicCounter = Option(termTopicCounter).map(_.join(newTermTopicCounter).map {
        case (term, (a, b)) =>
          (term, a :+ b)
      }).getOrElse(newTermTopicCounter)

      termTopicCounter.persist(storageLevel).count()
      Option(previousTermTopicCounter).foreach(_.unpersist(blocking = false))
      previousTermTopicCounter = termTopicCounter
    }
    val rand = new Random()
    val ttc = termTopicCounter.mapValues(c => {
      val nc = new BSV[Count](c.index.slice(0, c.used), c.data.slice(0, c.used).map(v => {
        val mid = v.toDouble / saveIter
        val l = math.floor(mid)
        if (rand.nextDouble() > mid - l) l else l + 1
      }.toInt), c.length)
      nc
    })
    ttc.persist(storageLevel)
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
      val mergingCorpus = corpus.mapEdges(edges => {
        val topics = edges.attr
        val newTopics = edges.attr.map { topic =>
          minMap.getOrElse(topic, topic)
        }
        (topics, newTopics)
      })
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
    var totalProb = 0D

    // \frac{{\alpha }_{k}{\beta }_{w}}{{n}_{k}+\bar{\beta }}
    totalTopicCounter.activeIterator.foreach { case (topic, cn) =>
      totalProb += alpha * beta / (cn + numTerms * beta)
    }

    val termProb = corpus.mapVertices { (vid, counter) =>
      val probDist = BSV.zeros[Double](numTopics)
      if (vid >= 0) {
        val termTopicCounter = counter
        // \frac{{n}_{kw}{\alpha }_{k}}{{n}_{k}+\bar{\beta }}
        termTopicCounter.activeIterator.foreach { case (topic, cn) =>
          probDist(topic) = cn * alpha /
            (totalTopicCounter(topic) + numTerms * beta)
        }
      } else {
        val docTopicCounter = counter
        // \frac{{n}_{kd}{\beta }_{w}}{{n}_{k}+\bar{\beta }}
        docTopicCounter.activeIterator.foreach { case (topic, cn) =>
          probDist(topic) = cn * beta /
            (totalTopicCounter(topic) + numTerms * beta)
        }
      }
      probDist.compact()
      (counter, probDist)
    }.mapTriplets { triplet =>
      val (termTopicCounter, termProb) = triplet.srcAttr
      val (docTopicCounter, docProb) = triplet.dstAttr
      val docSize = brzSum(docTopicCounter)
      val docTermSize = triplet.attr.length
      var prob = 0D

      // \frac{{n}_{kw}{n}_{kd}}{{n}_{k}+\bar{\beta}}
      docTopicCounter.activeIterator.foreach { case (topic, cn) =>
        prob += cn * termTopicCounter(topic) /
          (totalTopicCounter(topic) + numTerms * beta)
      }
      prob += brzSum(docProb) + brzSum(termProb) + totalProb
      prob += prob / (docSize + numTopics * alpha)

      docTermSize * Math.log(prob)
    }.edges.map(t => t.attr).sum()

    math.exp(-1 * termProb / numTokens)
  }

}

object LDA {
  private[ml] type DocId = VertexId
  private[ml] type WordId = VertexId
  private[ml] type Count = Int
  private[ml] type ED = Array[Int]
  private[ml] type VD = BSV[Count]
  private[ml] type BOW = (Long, BSV[Int])

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
   * @param LDAAlgorithm which LDA sampling algorithm to use, recommend not lightlda for short text
   * @param partStrategy which partition strategy to re partition by the graph
   * @param storageLevel StorageLevel that the LDA Model RDD uses
   * @return DistributedLDAModel
   */
  def train(
    docs: RDD[BOW],
    totalIter: Int,
    numTopics: Int,
    alpha: Float,
    beta: Float,
    alphaAS: Float,
    LDAAlgorithm: String,
    partStrategy: String,
    storageLevel: StorageLevel): DistributedLDAModel = {
    val lda: LDA = LDAAlgorithm match {
      case "lightlda" =>
        println("using LightLDA sampling algorithm.")
        LightLDA(docs, numTopics, alpha, beta, alphaAS, storageLevel, partStrategy)
      case "fastlda" =>
        println("using FastLDA sampling algorithm.")
        FastLDA(docs, numTopics, alpha, beta, alphaAS, storageLevel, partStrategy)
      case _ =>
        throw new NoSuchMethodException("No this algorithm or not implemented.")
    }
    lda.runGibbsSampling(totalIter)
    lda.saveModel(1)
  }

  def incrementalTrain(
    docs: RDD[BOW],
    computedModel: LocalLDAModel,
    totalIter: Int,
    LDAAlgorithm: String,
    partStrategy: String,
    storageLevel: StorageLevel): DistributedLDAModel = {
    val numTopics = computedModel.numTopics
    val alpha = computedModel.alpha
    val beta = computedModel.beta
    val alphaAS = computedModel.alphaAS
    val broadcastModel = docs.context.broadcast(computedModel)
    val lda: LDA = LDAAlgorithm match {
      case "lightlda" =>
        println("using LightLDA sampling algorithm.")
        LightLDA(docs, numTopics, alpha, beta, alphaAS, storageLevel, partStrategy, broadcastModel)
      case "fastlda" =>
        println("using FastLDA sampling algorithm.")
        FastLDA(docs, numTopics, alpha, beta, alphaAS, storageLevel, partStrategy, broadcastModel)
      case _ =>
        throw new NoSuchMethodException("No this algorithm or not implemented.")
    }
    broadcastModel.unpersist(blocking = false)
    lda.runGibbsSampling(totalIter)
    lda.saveModel(1)
  }

  private[ml] def initializeCorpus(
    docs: RDD[BOW],
    numTopics: Int,
    storageLevel: StorageLevel,
    partStrategy: String,
    computedModel: Broadcast[LocalLDAModel] = null): Graph[VD, ED] = {
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
    initCorpus.persist(storageLevel)
    initCorpus.vertices.setName("initVertices").count()
    val numEdges = initCorpus.edges.setName("initEdges").count()
    println(s"edges in the corpus: $numEdges")
    docs.unpersist(blocking = false)
    initCorpus = partStrategy match {
      case "dbh" =>
        DBHPartitioner.partitionByDBH[VD, ED](initCorpus, storageLevel)
      case "edge2d" =>
        initCorpus.partitionBy(PartitionStrategy.EdgePartition2D)
      case _ =>
        throw new NoSuchMethodException("No this algorithm or not implemented.")
    }
    val corpus = initCounter(initCorpus, numTopics)
    corpus.checkpoint()
    corpus.persist(storageLevel)
    corpus.vertices.setName("vertices")
    corpus.edges.setName("edges")
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
      docTopicCounter = computedModel.sampleTokens(docTopicCounter,
        tokens, topics)
    }
    doc.activeIterator.filter(_._2 > 0).map { case (term, counter) =>
      val ev = topics.zipWithIndex.filter { case (topic, offset) =>
        term == tokens(offset)
      }.map(_._1)
      Edge(term, newDocId, ev)
    }
  }

  // make docId always be negative, so that the doc vertex always be the dest vertex
  private def genNewDocId(docId: Long): Long = {
    assert(docId >= 0)
    -(docId + 1L)
  }

  private[ml] def sampleSV(
    gen: Random,
    table: AliasTable,
    sv: VD,
    currentTopic: Int,
    currentTopicCounter: Int = 0,
    numSampling: Int = 0): Int = {
    val docTopic = table.sampleAlias(gen)
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

  private def initCounter(initCorpus: Graph[VD, ED],
    numTopics: Int): Graph[VD, ED] = {
    val newCounter = initCorpus.edges.mapPartitions(iter => {
      var lastVid = -1L
      var lastTermSum: BSV[Count] = null
      val major = iter.flatMap(edge => {
        val vid = edge.srcId
        val did = edge.dstId
        val termSum = if (vid == lastVid) lastTermSum else BSV.zeros[Count](numTopics)
        var docSum = BSV.zeros[Count](numTopics)
        val topics = edge.attr
        for (t <- topics) {
          termSum(t) += 1
          docSum(t) += 1
        }
        if (vid == lastVid || lastVid == -1L) {
          Iterator.single((did, docSum))
        } else {
          val sendVid = lastVid
          val sendTermSum = lastTermSum
          lastVid = vid
          lastTermSum = termSum
          Iterator((sendVid, sendTermSum), (did, docSum))
        }
      })
      if (lastVid != -1L) {
        major ++ Iterator((lastVid, lastTermSum))
      } else {
        major
      }
    }).reduceByKey(initCorpus.vertices.partitioner.get, _ += _)
    GraphImpl(VertexRDD(newCounter), initCorpus.edges)
  }

  private def updateCounter(sampledCorpus: Graph[VD, (ED, ED)],
    numTopics: Int): Graph[VD, ED] = {
    val deltaCounter = sampledCorpus.edges.mapPartitions(iter => {
      var lastVid = -1L
      var lastTermDeltaSum: BSV[Count] = null
      val major = iter.flatMap(edge => {
        val vid = edge.srcId
        val did = edge.dstId
        val termDeltaSum = if (vid == lastVid) lastTermDeltaSum else BSV.zeros[Count](numTopics)
        var docDeltaSum = BSV.zeros[Count](numTopics)
        val prevTopics = edge.attr._1
        val newTopics = edge.attr._2
        for ((t, nt) <- prevTopics.zip(newTopics).filter(t => t._1 != t._2)) {
          termDeltaSum(t) -= 1
          termDeltaSum(nt) += 1
          docDeltaSum(t) -= 1
          docDeltaSum(nt) += 1
        }
        if (vid == lastVid || lastVid == -1L) {
          Iterator.single((did, docDeltaSum))
        } else {
          val sendVid = lastVid
          val sendTermDeltaSum = lastTermDeltaSum
          lastVid = vid
          lastTermDeltaSum = termDeltaSum
          Iterator((sendVid, sendTermDeltaSum), (did, docDeltaSum))
        }
      })
      if (lastVid != -1L) {
        major ++ Iterator((lastVid, lastTermDeltaSum))
      } else {
        major
      }
    }).reduceByKey(sampledCorpus.vertices.partitioner.get, _ += _)
    sampledCorpus.joinVertices(deltaCounter)((vid, counter, delta) => counter += delta)
      .mapEdges(edge => edge.attr._2)
  }
}

private[ml] class LDAKryoRegistrator extends KryoRegistrator {
  def registerClasses(kryo: com.esotericsoftware.kryo.Kryo) {
    kryo.register(classOf[BSV[LDA.Count]])
    kryo.register(classOf[BSV[Float]])

    kryo.register(classOf[BDV[LDA.Count]])
    kryo.register(classOf[BDV[Double]])

    kryo.register(classOf[LDA.ED])
    kryo.register(classOf[LDA.VD])
    kryo.register(classOf[LDA.BOW])

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
  alpha: Float,
  beta: Float,
  alphaAS: Float,
  storageLevel: StorageLevel)
  extends LDA(corpus, numTopics, numTerms, numDocs, numTokens, alpha, beta, alphaAS, storageLevel) {

  override protected def sampleTokens(pseudoIter: Int): Graph[VD, (ED, ED)] = {
    val numPartitions = corpus.edges.partitions.length
    val nweGraph = corpus.mapTriplets((pid, iter) => {
      val gen = new XORShiftRandom(numPartitions * pseudoIter + pid)
      // table is a per term data structure
      // in GraphX, edges in a partition are clustered by source IDs (term id in this case)
      // so, use below simple cache to avoid calculating table each time
      val lastTable = new AliasTable(numTopics)
      var lastVid: VertexId = -1L
      var lastWSum = 0F
      val dv = tDense()
      val dData = new Array[Float](numTopics)
      val t = AliasTable.generateAlias(dv._2, dv._1)
      val tSum = dv._1
      iter.map {
        triplet =>
          val termId = triplet.srcId
          val docId = triplet.dstId
          val termTopicCounter = triplet.srcAttr
          val docTopicCounter = triplet.dstAttr
          val topics = triplet.attr
          val newTopics = topics.clone()
          for (i <- topics.indices) {
            val currentTopic = topics(i)
            dSparse(termTopicCounter, docTopicCounter, dData, currentTopic)
            if (lastVid != termId) {
              lastWSum = wordTable(lastTable, termTopicCounter)
              lastVid = termId
            }
            val newTopic = tokenSampling(gen, t, tSum, lastTable, termTopicCounter, lastWSum,
              docTopicCounter, dData, currentTopic)

            if (newTopic != currentTopic) {
              newTopics(i) = newTopic
            }
          }

          (topics, newTopics)
      }
    }, TripletFields.All)
    GraphImpl(nweGraph.vertices.mapValues(t => null), nweGraph.edges)
  }

  private def tokenSampling(
    gen: Random,
    t: AliasTable,
    tSum: Float,
    w: AliasTable,
    termTopicCounter: VD,
    wSum: Float,
    docTopicCounter: VD,
    dData: Array[Float],
    currentTopic: Int): Int = {
    val index = docTopicCounter.index
    val used = docTopicCounter.used
    val dSum = dData(docTopicCounter.used - 1)
    val distSum = tSum + wSum + dSum
    val genSum = gen.nextFloat() * distSum
    if (genSum < dSum) {
      val dGenSum = gen.nextFloat() * dSum
      val pos = binarySearchInterval[Float](dData, dGenSum, 0, used, greater=true)
      index(pos)
    } else if (genSum < (dSum + wSum)) {
      sampleSV(gen, w, termTopicCounter, currentTopic)
    } else {
      t.sampleAlias(gen)
    }
  }

  /**
   * dense part in the decomposed sampling formula:
   * t = \frac{{\beta }_{w} \bar{\alpha} ( {n}_{k}^{-di} + \acute{\alpha} ) } {({n}_{k}^{-di}+\bar{\beta})
   * ({\sum{n}_{k}^{-di} +\bar{\acute{\alpha}}})}
   */
  private def tDense(): (Float, BDV[Float]) = {
    val t = BDV.zeros[Float](numTopics)
    val alphaSum = alpha * numTopics
    val termSum = numTokens - 1F + alphaAS * numTopics
    val betaSum = numTerms * beta
    var sum = 0F
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
  private def wSparse(termTopicCounter: VD): (Float, BSV[Float]) = {
    val alphaSum = alpha * numTopics
    val termSum = numTokens - 1F + alphaAS * numTopics
    val betaSum = numTerms * beta
    val w = BSV.zeros[Float](numTopics)
    var sum = 0F
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
  private def dSparse(
    termTopicCounter: VD,
    docTopicCounter: VD,
    d: Array[Float],
    currentTopic: Int): Unit = {
    val index = docTopicCounter.index
    val data = docTopicCounter.data
    val used = docTopicCounter.used

    // val termSum = numTokens - 1D + alphaAS * numTopics
    val betaSum = numTerms * beta
    var sum = 0F
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

  private def wordTable(
    table: AliasTable,
    termTopicCounter: VD): Float = {
    val sv = wSparse(termTopicCounter)
    AliasTable.generateAlias(sv._2, sv._1, table)
    sv._1
  }
}

class LightLDA(
  corpus: Graph[VD, ED],
  numTopics: Int,
  numTerms: Int,
  numDocs: Long,
  numTokens: Long,
  alpha: Float,
  beta: Float,
  alphaAS: Float,
  storageLevel: StorageLevel)
  extends LDA(corpus, numTopics, numTerms, numDocs, numTokens, alpha, beta, alphaAS, storageLevel) {

  override protected def sampleTokens(pseudoIter: Int): Graph[VD, (ED, ED)] = {
    val numPartitions = corpus.edges.partitions.length
    val nweGraph = corpus.mapTriplets((pid, iter) => {
      val gen = new Random(numPartitions * pseudoIter + pid)
      val docTableCache = new AppendOnlyMap[VertexId, SoftReference[(Float, AliasTable)]]()

      // table is a per term data structure
      // in GraphX, edges in a partition are clustered by source IDs (term id in this case)
      // so, use below simple cache to avoid calculating table each time
      val lastTable = new AliasTable(numTopics.toInt)
      var lastVid: VertexId = -1L
      var lastWSum = 0F

      val p = tokenTopicProb _
      val dPFun = docProb _
      val wPFun = wordProb _

      var dD: AliasTable = null
      var dDSum: Float = 0F
      var wD: AliasTable = null
      var wDSum: Float = 0F

      iter.map {
        triplet =>
          val termId = triplet.srcId
          val docId = triplet.dstId
          val termTopicCounter = triplet.srcAttr
          val docTopicCounter = triplet.dstAttr
          val topics = triplet.attr
          val newTopics = topics.clone()

          if (dD == null || gen.nextDouble() < 1e-6) {
            var dv = dDense()
            dDSum = dv._1
            dD = AliasTable.generateAlias(dv._2, dDSum)

            dv = wDense()
            wDSum = dv._1
            wD = AliasTable.generateAlias(dv._2, wDSum)
          }
          val (dSum, d) = docTopicCounter.synchronized {
            docTable(x => x == null || x.get() == null || gen.nextDouble() < 1e-2,
              docTableCache, docTopicCounter, docId)
          }
          val (wSum, w) = termTopicCounter.synchronized {
            if (lastVid != termId || gen.nextDouble() < 1e-4) {
              lastWSum = wordTable(lastTable, termTopicCounter)
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
                if (gen.nextFloat() < dDSum / (dSum - 1F + dDSum)) {
                  proposalTopic = dD.sampleAlias(gen)
                }
                else {
                  proposalTopic = docTopicCounter.synchronized {
                    sampleSV(gen, d, docTopicCounter, currentTopic)
                  }
                }
                dPFun
              } else {
                val table = if (gen.nextDouble() < wSum / (wSum + wDSum)) w else wD
                proposalTopic = table.sampleAlias(gen)
                wPFun
              }

              val newTopic = tokenSampling(gen, docTopicCounter, termTopicCounter, docProposal,
                    currentTopic, proposalTopic, q, p)
              assert(newTopic >= 0 && newTopic < numTopics)
              if (newTopic != currentTopic) {
                newTopics(i) = newTopic
              }
            }
          }
          (topics, newTopics)
      }
    }, TripletFields.All)
    GraphImpl(nweGraph.vertices.mapValues(t => null), nweGraph.edges)
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
  def tokenSampling(
    gen: Random,
    docTopicCounter: VD,
    termTopicCounter: VD,
    docProposal: Boolean,
    currentTopic: Int,
    proposalTopic: Int,
    q: (VD, Int, Boolean) => Float,
    p: (VD, VD, Int, Boolean) => Float): Int = {
    if (proposalTopic == currentTopic) return proposalTopic
    val cp = p(docTopicCounter, termTopicCounter, currentTopic, true)
    val np = p(docTopicCounter, termTopicCounter, proposalTopic, false)
    val vd = if (docProposal) docTopicCounter else termTopicCounter
    val cq = q(vd, currentTopic, true)
    val nq = q(vd, proposalTopic, false)

    val pi = (np * cq) / (cp * nq)
    if (gen.nextFloat() < math.min(1F, pi)) proposalTopic else currentTopic
  }

  // scalastyle:off
  private def tokenTopicProb(docTopicCounter: VD,
    termTopicCounter: VD,
    topic: Int,
    isAdjustment: Boolean): Float = {
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

  private def wordProb(termTopicCounter: VD, topic: Int, isAdjustment: Boolean): Float = {
    (termTopicCounter(topic) + beta) / (totalTopicCounter(topic) + beta * numTerms)
  }

  private def docProb(docTopicCounter: VD, topic: Int, isAdjustment: Boolean): Float = {
    val adjustment = if (isAdjustment) -1 else 0
    val numTopics = totalTopicCounter.length
    val ratio = (totalTopicCounter(topic) + alphaAS) /
      (numTokens - 1 + alphaAS * numTopics)
    val asPrior = ratio * (alpha * numTopics)
    docTopicCounter(topic) + adjustment + asPrior
  }

  /**
   * \frac{{n}_{kw}}{{n}_{k}+\bar{\beta}}
   */
  private def wSparse(termTopicCounter: VD): (Float, BV[Float]) = {
    val termSum = beta * numTerms
    val w = BSV.zeros[Float](numTopics)
    var sum = 0F
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
  private def wDense(): (Float, BV[Float]) = {
    val t = BDV.zeros[Float](numTopics)
    val termSum = beta * numTerms
    var sum = 0F
    for (topic <- 0 until numTopics) {
      val last = beta / (totalTopicCounter(topic) + termSum)
      t(topic) = last
      sum += last
    }
    (sum, t)
  }

  private def dSparse(docTopicCounter: VD): (Float, BV[Float]) = {
    val d = BSV.zeros[Float](numTopics)
    var sum = 0F
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

  private def dDense(): (Float, BV[Float]) = {
    val asPrior = BDV.zeros[Float](numTopics)
    var sum = 0F
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
    updateFunc: SoftReference[(Float, AliasTable)] => Boolean,
    cacheMap: AppendOnlyMap[VertexId, SoftReference[(Float, AliasTable)]],
    docTopicCounter: VD,
    docId: VertexId): (Float, AliasTable) = {
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

  private def wordTable(
    table: AliasTable,
    termTopicCounter: VD): Float = {
    val sv = wSparse(termTopicCounter)
    AliasTable.generateAlias(sv._2, sv._1, table)
    sv._1
  }
}

object FastLDA {
  def apply(bowDocs: RDD[BOW],
    numTopics: Int,
    alpha: Float,
    beta: Float,
    alphaAS: Float,
    storageLevel: StorageLevel,
    partStrategy: String,
    computedModel: Broadcast[LocalLDAModel] = null) = {
    val numTerms = bowDocs.first()._2.size
    val numDocs = bowDocs.count()
    val corpus = initializeCorpus(bowDocs, numTopics, storageLevel, partStrategy, computedModel)
    val numTokens = corpus.edges.map(e => e.attr.length.toDouble).sum().toLong
    new FastLDA(corpus, numTopics, numTerms, numDocs, numTokens, alpha, beta, alphaAS, storageLevel)
  }
}

object LightLDA {
  def apply(bowDocs: RDD[BOW],
    numTopics: Int,
    alpha: Float,
    beta: Float,
    alphaAS: Float,
    storageLevel: StorageLevel,
    partStrategy: String,
    computedModel: Broadcast[LocalLDAModel] = null) = {
    val numTerms = bowDocs.first()._2.size
    val numDocs = bowDocs.count()
    val corpus = initializeCorpus(bowDocs, numTopics, storageLevel, partStrategy, computedModel)
    val numTokens = corpus.edges.map(e => e.attr.length.toDouble).sum().toLong
    new LightLDA(corpus, numTopics, numTerms, numDocs, numTokens, alpha, beta, alphaAS, storageLevel)
  }
}
