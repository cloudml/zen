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

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, sum => brzSum, Vector => BV, normalize => brzNormalize}
import com.github.cloudml.zen.ml.DBHPartitioner
import com.github.cloudml.zen.ml.clustering.LDA._
import com.github.cloudml.zen.ml.clustering.LDAUtils._
import com.github.cloudml.zen.ml.util.SparkUtils._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.graphx._
import org.apache.spark.graphx.impl.GraphImpl
import org.apache.spark.mllib.linalg.distributed.{MatrixEntry, RowMatrix}
import org.apache.spark.mllib.linalg.{DenseVector => SDV, SparseVector => SSV, Vector => SV}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.serializer.KryoRegistrator
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.collection.AppendOnlyMap
import org.apache.spark.{HashPartitioner, Logging}

abstract class LDA private[ml](
  @transient private var corpus: Graph[VD, ED],
  private val numTopics: Int,
  private val numTerms: Int,
  private var alpha: Double,
  private var beta: Double,
  private var alphaAS: Double,
  private var storageLevel: StorageLevel) extends Serializable with Logging {

  /**
   * 语料库文档数
   */
  val numDocs = docVertices.count()

  /**
   * 语料库总的词数(包含重复)
   */
  val numTokens = corpus.edges.map(e => e.attr.size.toDouble).sum().toLong

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

  def setStorageLevel(newStorageLevel: StorageLevel): this.type = {
    this.storageLevel = newStorageLevel
    this
  }

  def setSeed(newSeed: Int): this.type = {
    this.seed = newSeed
    this
  }

  def getCorpus = corpus

  @transient private var seed = new Random().nextInt()
  @transient private var innerIter = 1
  @transient private var totalTopicCounter: BDV[Count] = collectTotalTopicCounter(corpus)

  private def termVertices = corpus.vertices.filter(t => t._1 >= 0)

  private def docVertices = corpus.vertices.filter(t => t._1 < 0)

  private def checkpoint(corpus: Graph[VD, ED]): Unit = {
    val sc = corpus.edges.sparkContext
    if (innerIter % 10 == 0 && sc.getCheckpointDir.isDefined) {
      corpus.checkpoint()
    }
  }

  private def collectTotalTopicCounter(graph: Graph[VD, ED]): BDV[Count] = {
    val globalTopicCounter = collectGlobalCounter(graph, numTopics)
    assert(brzSum(globalTopicCounter) == numTokens)
    globalTopicCounter
  }

  private def gibbsSampling(iter: Int): Unit = {
    val previousCorpus = corpus
    val sampledCorpus = sampleTokens(corpus, totalTopicCounter, innerIter + seed,
      numTokens, numTopics, numTerms, alpha, alphaAS, beta)
    sampledCorpus.persist(storageLevel)
    sampledCorpus.edges.setName(s"sampledEdges-$iter")
    sampledCorpus.vertices.setName(s"sampledVertices-$iter")

    corpus = updateCounter(sampledCorpus, numTopics)
    checkpoint(corpus)
    corpus.persist(storageLevel)
    corpus.edges.setName(s"edges-$iter").count()
    corpus.vertices.setName(s"vertices-$iter")
    // completeCorpus.vertices.count()
    totalTopicCounter = collectTotalTopicCounter(corpus)

    previousCorpus.edges.unpersist(blocking = false)
    previousCorpus.vertices.unpersist(blocking = false)
    sampledCorpus.edges.unpersist(blocking = false)
    sampledCorpus.vertices.unpersist(blocking = false)
    innerIter += 1
  }

  private def collectGlobalCounter(graph: Graph[VD, ED], numTopics: Int): BDV[Count] = {
    graph.vertices.filter(t => t._1 >= 0).map(_._2).aggregate(BDV.zeros[Count](numTopics))(_ :+= _, _ :+= _)
  }

  protected def sampleTokens(
    graph: Graph[VD, ED],
    totalTopicCounter: BDV[Count],
    innerIter: Long,
    numTokens: Double,
    numTopics: Double,
    numTerms: Double,
    alpha: Double,
    alphaAS: Double,
    beta: Double): Graph[VD, ED]

  def saveModel(iter: Int = 1): LDAModel = {
    var termTopicCounter: RDD[(VertexId, VD)] = null
    for (iter <- 1 to iter) {
      logInfo(s"Save TopicModel (Iteration $iter/$iter)")
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
    val model = LDAModel(numTopics, numTerms, alpha, beta)
    termTopicCounter.collect().foreach { case (term, counter) =>
      model.merge(term.toInt, counter)
    }
    model.gtc :/= iter.toDouble
    model.ttc.foreach { ttc =>
      ttc :/= iter.toDouble
      ttc.compact()
    }
    model
  }

  def runGibbsSampling(iterations: Int): Unit = {
    for (iter <- 1 to iterations) {
      logInfo(s"Start Gibbs sampling (Iteration $iter/$iterations)")
      val startedAt = System.nanoTime()
      gibbsSampling(iter)
      val elapsedSeconds = (System.nanoTime() - startedAt) / 1e9
      // logInfo(s"Gibbs sampling (Iteration $iter/$iterations):  ${perplexity()}")
      logInfo(s"End Gibbs sampling  (Iteration $iter/$iterations) takes:  $elapsedSeconds")
    }
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
    if (minMap.size > 0) {
      corpus = corpus.mapEdges(edges => {
        edges.attr.map { topic =>
          minMap.get(topic).getOrElse(topic)
        }
      })
      corpus = updateCounter(corpus, numTopics)
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
    val numTopics = this.numTopics
    val numTerms = this.numTerms
    val alpha = this.alpha
    val beta = this.beta
    val totalSize = brzSum(totalTopicCounter)
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

    math.exp(-1 * termProb / totalSize)
  }

}

object LDA {

  private[ml] type DocId = VertexId
  private[ml] type WordId = VertexId
  private[ml] type Count = Int
  private[ml] type ED = Array[Count]
  private[ml] type VD = BSV[Count]

  /**
   * LDA training
   * @param docs training data (docId, wordId) docId >= 0, wordId < 0
   * @param numTopics the number of topics (5000+ for large data) 
   * @param totalIter  the number of iterations
   * @param alpha      recommend to be (5.0 /numTopics)
   * @param beta       recommend to be in range 0.001 - 0.1
   * @param alphaAS    recommend to be in range 0.01 - 1.0
   * @param useLightLDA use LightLDA sampling algorithm or not, recommend false for short doc
   * @return LDAModel
   */
  def train(
    docs: RDD[(Long, SV)],
    numTopics: Int = 2048,
    totalIter: Int = 150,
    alpha: Double = 0.001,
    beta: Double = 0.01,
    alphaAS: Double = 0.1,
    useLightLDA: Boolean = false): LDAModel = {
    require(totalIter > 0, "totalIter is less than 0")
    val lda = if (useLightLDA) {
      new LightLDA(docs, numTopics, alpha, beta, alphaAS)
    } else {
      new FastLDA(docs, numTopics, alpha, beta, alphaAS)
    }
    lda.runGibbsSampling(totalIter - 1)
    lda.saveModel(1)
  }

  /**
   * Train and save the trained model. The training data is in LibSVM format where each line is represented in:
   * topicID termID+1:counter termID+1:counter ..
   * @param docs training data: (docId, wordId) docId >= 0 while wordId < 0
   * @param dir   the directory where the model is saved
   * @param numTopics  the number of topics (> 5000 for large data)
   * @param totalIter  the number of iterations
   * @param alpha      recommend to be (5.0 /numTopics)
   * @param beta       recommend to be in range 0.001 - 0.1
   * @param alphaAS    recommend to be in range 0.01 - 1.0
   * @param useLightLDA use LightLDA or not? Recommend to be false for short length document
   * @return LDAModel
   */
  def trainAndSaveModel(
    docs: RDD[(Long, SV)],
    dir: String,
    numTopics: Int = 2048,
    totalIter: Int = 150,
    alpha: Double = 0.01,
    beta: Double = 0.01,
    alphaAS: Double = 0.1,
    useLightLDA: Boolean = false): Unit = {
    val lda = if (useLightLDA) {
      new LightLDA(docs, numTopics, alpha, beta, alphaAS)
    } else {
      new FastLDA(docs, numTopics, alpha, beta, alphaAS)
    }
    val numTerms = lda.numTerms
    lda.runGibbsSampling(totalIter)
    val rdd = lda.termVertices.flatMap { case (termId, counter) =>
      counter.activeIterator.map { case (topic, cn) =>
        val sv = BSV.zeros[Double](numTerms)
        sv(termId.toInt) = cn.toDouble
        (topic, sv)
      }
    }.reduceByKey { (a, b) => a + b }.map { case (topic, sv) =>
      LabeledPoint(topic.toDouble, fromBreeze(sv))
    }
    MLUtils.saveAsLibSVMFile(rdd, dir)
  }

  /**
   * 增量训练
   * @param docs
   * @param computedModel
   * @param alphaAS
   * @param totalIter
   * @param useLightLDA
   * @return
   */
  def incrementalTrain(
    docs: RDD[(Long, SV)],
    computedModel: LDAModel,
    alphaAS: Double = 0.1,
    totalIter: Int = 150,
    useLightLDA: Boolean = false): LDAModel = {
    require(totalIter > 0, "totalIter is less than 0")
    val numTopics = computedModel.ttc.size
    val alpha = computedModel.alpha
    val beta = computedModel.beta
    val broadcastModel = docs.context.broadcast(computedModel)
    val lda = if (useLightLDA) {
      new LightLDA(docs, numTopics, alpha, beta, alphaAS, computedModel = broadcastModel)
    } else {
      new FastLDA(docs, numTopics, alpha, beta, alphaAS, computedModel = broadcastModel)
    }
    broadcastModel.unpersist(blocking = false)
    lda.runGibbsSampling(totalIter - 1)
    lda.saveModel(1)
  }

  private[ml] def initializeCorpus(
    docs: RDD[(LDA.DocId, SV)],
    numTopics: Int,
    storageLevel: StorageLevel,
    computedModel: Broadcast[LDAModel] = null): Graph[VD, ED] = {
    val edges = docs.mapPartitionsWithIndex((pid, iter) => {
      val gen = new Random(pid + 117)
      iter.flatMap { case (docId, doc) =>
        if (computedModel == null) {
          initializeEdges(gen, doc, docId, numTopics)
        } else {
          initEdgesWithComputedModel(gen, doc, docId, numTopics, computedModel.value)
        }
      }
    })
    edges.persist(storageLevel)
    var corpus: Graph[VD, ED] = Graph.fromEdges(edges, null, storageLevel, storageLevel)
    corpus.persist(storageLevel)
    corpus.vertices.count()
    corpus.edges.count()
    edges.unpersist(blocking = false)
    corpus = partitionByDBH(corpus, storageLevel)
    updateCounter(corpus, numTopics)
  }

  private def partitionByDBH(corpus: Graph[VD, ED], storageLevel: StorageLevel): Graph[VD, ED] = {
    val edges = corpus.edges
    val degrees = corpus.degrees.setName("degrees").persist(storageLevel)
    val degreesGraph = GraphImpl(degrees, edges)
    degreesGraph.persist(storageLevel)
    val numPartitions = edges.partitions.size
    val partitionStrategy = new DBHPartitioner(numPartitions, 0)
    val newEdges = degreesGraph.triplets.mapPartitions { iter =>
      iter.map { e =>
        (partitionStrategy.getPartition(e), Edge(e.srcId, e.dstId, e.attr))
      }
    }.partitionBy(new HashPartitioner(numPartitions)).map(_._2).persist(storageLevel)
    val dataSet = Graph.fromEdges(newEdges, null.asInstanceOf[VD], storageLevel, storageLevel)
    dataSet.persist(storageLevel)
    dataSet.vertices.count()
    dataSet.edges.count()
    degrees.unpersist(blocking = false)
    degreesGraph.vertices.unpersist(blocking = false)
    degreesGraph.edges.unpersist(blocking = false)
    corpus.vertices.unpersist(blocking = false)
    corpus.edges.unpersist(blocking = false)
    newEdges.unpersist(blocking = false)
    dataSet
  }

  private def initializeEdges(
    gen: Random,
    doc: SV,
    docId: DocId,
    numTopics: Int): Iterator[Edge[ED]] = {
    assert(docId >= 0)
    val newDocId: DocId = genNewDocId(docId)
    doc.activeIterator.filter(_._2 > 0).map { case (termId, counter) =>
      val topics = new Array[Int](counter.toInt)
      for (i <- 0 until counter.toInt) {
        topics(i) = gen.nextInt(numTopics)
      }
      Edge(termId, newDocId, topics)
    }
  }

  private def initEdgesWithComputedModel(
    gen: Random,
    doc: SV,
    docId: DocId,
    numTopics: Int,
    computedModel: LDAModel = null): Iterator[Edge[ED]] = {
    assert(docId >= 0)
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

  private def genNewDocId(docId: Long): Long = {
    -(docId + 1L)
  }

  private[ml] def sampleSV(gen: Random, table: Table, sv: VD, currentTopic: Int): Int = {
    val docTopic = sampleAlias(gen, table)
    if (docTopic == currentTopic) {
      val svCounter = sv(currentTopic)
      // TODO: not sure it is correct or not?
      // discard it if the newly sampled topic is current topic
      if ((svCounter == 1 && table._1.length > 1) ||
        /* the sampled topic that contains current token and other tokens */
        (svCounter > 1 && gen.nextDouble() < 1.0 / svCounter)
        /* the sampled topic has 1/svCounter probability that belongs to current token */) {
        return sampleSV(gen, table, sv, currentTopic)
      }
    }
    docTopic
  }

  private def updateCounter(graph: Graph[_, ED], numTopics: Int): Graph[VD, ED] = {
    val newCounter = graph.aggregateMessages[VD](ctx => {
      val topics = ctx.attr
      val vector = BSV.zeros[Count](numTopics)
      for (topic <- topics) {
        vector(topic) += 1
      }
      ctx.sendToDst(vector)
      ctx.sendToSrc(vector)
    }, _ + _, TripletFields.EdgeOnly).mapValues(v => {
      val used = v.used
      if (v.index.length == used) {
        v
      } else {
        val index = new Array[Int](used)
        val data = new Array[Count](used)
        Array.copy(v.index, 0, index, 0, used)
        Array.copy(v.data, 0, data, 0, used)
        new BSV[Count](index, data, numTopics)
      }
    })
    // GraphImpl.fromExistingRDDs(newCounter, graph.edges)
    GraphImpl(newCounter, graph.edges)
  }
}

private[ml] class LDAKryoRegistrator extends KryoRegistrator {
  def registerClasses(kryo: com.esotericsoftware.kryo.Kryo) {
    val gkr = new GraphKryoRegistrator
    gkr.registerClasses(kryo)

    kryo.register(classOf[BSV[LDA.Count]])
    kryo.register(classOf[BSV[Double]])

    kryo.register(classOf[BDV[LDA.Count]])
    kryo.register(classOf[BDV[Double]])

    kryo.register(classOf[SV])
    kryo.register(classOf[SSV])
    kryo.register(classOf[SDV])

    kryo.register(classOf[LDA.ED])
    kryo.register(classOf[LDA.VD])

    kryo.register(classOf[Random])
    kryo.register(classOf[LDA])
    kryo.register(classOf[LDAModel])
  }
}

class FastLDA(
  corpus: Graph[VD, ED],
  numTopics: Int,
  numTerms: Int,
  alpha: Double,
  beta: Double,
  alphaAS: Double,
  storageLevel: StorageLevel) extends LDA(corpus, numTopics, numTerms, alpha, beta, alphaAS, storageLevel) {
  def this(docs: RDD[(DocId, SV)],
    numTopics: Int,
    alpha: Double,
    beta: Double,
    alphaAS: Double,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK,
    computedModel: Broadcast[LDAModel] = null) {
    this(initializeCorpus(docs, numTopics, storageLevel, computedModel),
      numTopics, docs.first()._2.size, alpha, beta, alphaAS, storageLevel)
  }

  override protected def sampleTokens(
    graph: Graph[VD, ED],
    totalTopicCounter: BDV[Count],
    innerIter: Long,
    numTokens: Double,
    numTopics: Double,
    numTerms: Double,
    alpha: Double,
    alphaAS: Double,
    beta: Double): Graph[VD, ED] = {
    val parts = graph.edges.partitions.size
    val nweGraph = graph.mapTriplets(
      (pid, iter) => {
        val gen = new Random(parts * innerIter + pid)
        val wordTableCache = new AppendOnlyMap[VertexId, SoftReference[(Double, Table)]]()
        val dv = tDense(totalTopicCounter, numTokens, numTerms, alpha, alphaAS, beta)
        val dData = new Array[Double](numTopics.toInt)
        val t = generateAlias(dv._2, dv._1)
        val tSum = dv._1
        iter.map {
          triplet =>
            val termId = triplet.srcId
            val docId = triplet.dstId
            val termTopicCounter = triplet.srcAttr
            val docTopicCounter = triplet.dstAttr
            val topics = triplet.attr
            for (i <- 0 until topics.length) {
              val currentTopic = topics(i)
              dSparse(totalTopicCounter, termTopicCounter, docTopicCounter, dData,
                currentTopic, numTokens, numTerms, alpha, alphaAS, beta)
              val (wSum, w) = wordTable(x => x == null || x.get() == null,
                wordTableCache, totalTopicCounter, termTopicCounter,
                termId, numTokens, numTerms, alpha, alphaAS, beta)
              val newTopic = tokenSampling(gen, t, tSum, w, termTopicCounter, wSum,
                docTopicCounter, dData, currentTopic)

              if (newTopic != currentTopic) {
                topics(i) = newTopic
              }
            }

            topics
        }
      }, TripletFields.All)
    GraphImpl(nweGraph.vertices.mapValues(t => null), nweGraph.edges)
  }

  private def tokenSampling(
    gen: Random,
    t: Table,
    tSum: Double,
    w: Table,
    termTopicCounter: VD,
    wSum: Double,
    docTopicCounter: VD,
    dData: Array[Double],
    currentTopic: Int): Int = {
    val index = docTopicCounter.index
    val used = docTopicCounter.used
    val dSum = dData(docTopicCounter.used - 1)
    val distSum = tSum + wSum + dSum
    val genSum = gen.nextDouble() * distSum
    if (genSum < dSum) {
      val dGenSum = gen.nextDouble() * dSum
      val pos = binarySearchInterval(dData, dGenSum, 0, used, true)
      index(pos)
    } else if (genSum < (dSum + wSum)) {
      sampleSV(gen, w, termTopicCounter, currentTopic)
    } else {
      sampleAlias(gen, t)
    }
  }

  /**
   * dense part in the decomposed sampling formula:
   * t = \frac{{\beta }_{w} \bar{\alpha} ( {n}_{k}^{-di} + \acute{\alpha} ) } {({n}_{k}^{-di}+\bar{\beta})
   * ({\sum{n}_{k}^{-di} +\bar{\acute{\alpha}}})}
   */
  private def tDense(
    totalTopicCounter: BDV[Count],
    numTokens: Double,
    numTerms: Double,
    alpha: Double,
    alphaAS: Double,
    beta: Double): (Double, BDV[Double]) = {
    val numTopics = totalTopicCounter.length
    val t = BDV.zeros[Double](numTopics)
    val alphaSum = alpha * numTopics
    val termSum = numTokens - 1D + alphaAS * numTopics
    val betaSum = numTerms * beta
    var sum = 0.0
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
  private def wSparse(
    totalTopicCounter: BDV[Count],
    termTopicCounter: VD,
    numTokens: Double,
    numTerms: Double,
    alpha: Double,
    alphaAS: Double,
    beta: Double): (Double, BSV[Double]) = {
    val numTopics = totalTopicCounter.length
    val alphaSum = alpha * numTopics
    val termSum = numTokens - 1D + alphaAS * numTopics
    val betaSum = numTerms * beta
    val w = BSV.zeros[Double](numTopics)
    var sum = 0.0
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
    totalTopicCounter: BDV[Count],
    termTopicCounter: VD,
    docTopicCounter: VD,
    d: Array[Double],
    currentTopic: Int,
    numTokens: Double,
    numTerms: Double,
    alpha: Double,
    alphaAS: Double,
    beta: Double): Unit = {
    val index = docTopicCounter.index
    val data = docTopicCounter.data
    val used = docTopicCounter.used

    // val termSum = numTokens - 1D + alphaAS * numTopics
    val betaSum = numTerms * beta
    var sum = 0.0
    for (i <- 0 until used) {
      val topic = index(i)
      val count = data(i)
      val adjustment = if (currentTopic == topic) -1D else 0
      val last = (count + adjustment) * (termTopicCounter(topic) + adjustment + beta) /
        (totalTopicCounter(topic) + adjustment + betaSum)
      // val lastD = (count + adjustment) * termSum * (termTopicCounter(topic) + adjustment + beta) /
      //  ((totalTopicCounter(topic) + adjustment + betaSum) * termSum)
      sum += last
      d(i) = sum
    }
  }

  private def wordTable(
    updateFunc: SoftReference[(Double, Table)] => Boolean,
    cacheMap: AppendOnlyMap[VertexId, SoftReference[(Double, Table)]],
    totalTopicCounter: BDV[Count],
    termTopicCounter: VD,
    termId: VertexId,
    numTokens: Double,
    numTerms: Double,
    alpha: Double,
    alphaAS: Double,
    beta: Double): (Double, Table) = {
    val cacheW = cacheMap(termId)
    if (!updateFunc(cacheW)) {
      cacheW.get
    } else {
      val sv = wSparse(totalTopicCounter, termTopicCounter,
        numTokens, numTerms, alpha, alphaAS, beta)
      val w = (sv._1, generateAlias(sv._2, sv._1))
      cacheMap.update(termId, new SoftReference(w))
      w
    }
  }
}

class LightLDA(
  corpus: Graph[VD, ED],
  numTopics: Int,
  numTerms: Int,
  alpha: Double,
  beta: Double,
  alphaAS: Double,
  storageLevel: StorageLevel) extends LDA(corpus, numTopics, numTerms, alpha, beta, alphaAS, storageLevel) {
  def this(docs: RDD[(Long, SV)],
    numTopics: Int,
    alpha: Double,
    beta: Double,
    alphaAS: Double,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK,
    computedModel: Broadcast[LDAModel] = null) {
    this(initializeCorpus(docs, numTopics, storageLevel, computedModel),
      numTopics, docs.first()._2.size, alpha, beta, alphaAS, storageLevel)
  }

  override protected def sampleTokens(
    graph: Graph[VD, ED],
    totalTopicCounter: BDV[Count],
    innerIter: Long,
    numTokens: Double,
    numTopics: Double,
    numTerms: Double,
    alpha: Double,
    alphaAS: Double,
    beta: Double): Graph[VD, ED] = {
    val parts = graph.edges.partitions.size
    val nweGraph = graph.mapTriplets(
      (pid, iter) => {
        val gen = new Random(parts * innerIter + pid)
        val docTableCache = new AppendOnlyMap[VertexId, SoftReference[(Double, Table)]]()
        val wordTableCache = new AppendOnlyMap[VertexId, SoftReference[(Double, Table)]]()
        val p = tokenTopicProb(totalTopicCounter, beta, alpha,
          alphaAS, numTokens, numTerms) _
        val dPFun = docProb(totalTopicCounter, alpha, alphaAS, numTokens) _
        val wPFun = wordProb(totalTopicCounter, numTerms, beta) _

        var dD: Table = null
        var dDSum: Double = 0.0
        var wD: Table = null
        var wDSum: Double = 0.0

        iter.map {
          triplet =>
            val termId = triplet.srcId
            val docId = triplet.dstId
            val termTopicCounter = triplet.srcAttr
            val docTopicCounter = triplet.dstAttr
            val topics = triplet.attr

            if (dD == null || gen.nextDouble() < 1e-6) {
              var dv = dDense(totalTopicCounter, alpha, alphaAS, numTokens)
              dDSum = dv._1
              dD = generateAlias(dv._2, dDSum)

              dv = wDense(totalTopicCounter, numTerms, beta)
              wDSum = dv._1
              wD = generateAlias(dv._2, wDSum)
            }
            val (dSum, d) = docTopicCounter.synchronized {
              docTable(x => x == null || x.get() == null || gen.nextDouble() < 1e-2,
                docTableCache, docTopicCounter, docId)
            }
            val (wSum, w) = termTopicCounter.synchronized {
              wordTable(x => x == null || x.get() == null || gen.nextDouble() < 1e-4,
                wordTableCache, totalTopicCounter, termTopicCounter, termId, numTerms, beta)
            }
            for (i <- 0 until topics.length) {
              var docProposal = gen.nextDouble() < 0.5
              var maxSampling = 8
              while (maxSampling > 0) {
                maxSampling -= 1
                docProposal = !docProposal
                val currentTopic = topics(i)
                var proposalTopic = -1
                val q = if (docProposal) {
                  if (gen.nextDouble() < dDSum / (dSum - 1.0 + dDSum)) {
                    proposalTopic = sampleAlias(gen, dD)
                  }
                  else {
                    proposalTopic = docTopicCounter.synchronized {
                      sampleSV(gen, d, docTopicCounter, currentTopic)
                    }
                  }
                  dPFun
                } else {
                  val table = if (gen.nextDouble() < wSum / (wSum + wDSum)) w else wD
                  proposalTopic = sampleAlias(gen, table)
                  wPFun
                }

                val newTopic = docTopicCounter.synchronized {
                  termTopicCounter.synchronized {
                    tokenSampling(gen, docTopicCounter, termTopicCounter, docProposal,
                      currentTopic, proposalTopic, q, p)
                  }
                }

                assert(newTopic >= 0 && newTopic < numTopics)
                if (newTopic != currentTopic) {
                  topics(i) = newTopic
                  docTopicCounter.synchronized {
                    docTopicCounter(currentTopic) -= 1
                    docTopicCounter(newTopic) += 1
                  }
                  termTopicCounter.synchronized {
                    termTopicCounter(currentTopic) -= 1
                    termTopicCounter(newTopic) += 1
                  }
                  totalTopicCounter(currentTopic) -= 1
                  totalTopicCounter(newTopic) += 1
                }
              }
            }
            topics
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
    q: (VD, Int, Boolean) => Double,
    p: (VD, VD, Int, Boolean) => Double): Int = {
    if (proposalTopic == currentTopic) return proposalTopic
    val cp = p(docTopicCounter, termTopicCounter, currentTopic, true)
    val np = p(docTopicCounter, termTopicCounter, proposalTopic, false)
    val vd = if (docProposal) docTopicCounter else termTopicCounter
    val cq = q(vd, currentTopic, true)
    val nq = q(vd, proposalTopic, false)

    val pi = (np * cq) / (cp * nq)
    if (gen.nextDouble() < 1e-32) {
      println(s"Pi: ${pi}")
      println(s"($np * $cq) / ($cp * $nq)")
    }

    if (gen.nextDouble() < math.min(1.0, pi)) proposalTopic else currentTopic
  }

  // scalastyle:off
  private def tokenTopicProb(
    totalTopicCounter: BDV[Count],
    beta: Double,
    alpha: Double,
    alphaAS: Double,
    numTokens: Double,
    numTerms: Double)(docTopicCounter: VD,
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

  private def wordProb(
    totalTopicCounter: BDV[Count],
    numTerms: Double,
    beta: Double)(termTopicCounter: VD, topic: Int, isAdjustment: Boolean): Double = {
    (termTopicCounter(topic) + beta) / (totalTopicCounter(topic) + beta * numTerms)
  }

  private def docProb(
    totalTopicCounter: BDV[Count],
    alpha: Double,
    alphaAS: Double,
    numTokens: Double)(docTopicCounter: VD, topic: Int, isAdjustment: Boolean): Double = {
    val adjustment = if (isAdjustment) -1.0 else 0.0
    val numTopics = totalTopicCounter.length
    val ratio = (totalTopicCounter(topic) + alphaAS) /
      (numTokens - 1 + alphaAS * numTopics)
    val asPrior = ratio * (alpha * numTopics)
    docTopicCounter(topic) + adjustment + asPrior
  }

  /**
   * \frac{{n}_{kw}}{{n}_{k}+\bar{\beta}}
   */
  private def wSparse(
    totalTopicCounter: BDV[Count],
    termTopicCounter: VD,
    numTerms: Double,
    beta: Double): (Double, BV[Double]) = {
    val numTopics = termTopicCounter.length
    val termSum = beta * numTerms
    val w = BSV.zeros[Double](numTopics)

    var sum = 0.0
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
  private def wDense(
    totalTopicCounter: BDV[Count],
    numTerms: Double,
    beta: Double): (Double, BV[Double]) = {
    val numTopics = totalTopicCounter.length
    val t = BDV.zeros[Double](numTopics)
    val termSum = beta * numTerms
    var sum = 0.0
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
    var sum = 0.0
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


  private def dDense(
    totalTopicCounter: BDV[Count],
    alpha: Double,
    alphaAS: Double,
    numTokens: Double): (Double, BV[Double]) = {
    val numTopics = totalTopicCounter.length
    val asPrior = BDV.zeros[Double](numTopics)

    var sum = 0.0
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
    updateFunc: SoftReference[(Double, Table)] => Boolean,
    cacheMap: AppendOnlyMap[VertexId, SoftReference[(Double, Table)]],
    docTopicCounter: VD,
    docId: VertexId): (Double, Table) = {
    val cacheD = cacheMap(docId)
    if (!updateFunc(cacheD)) {
      cacheD.get
    } else {
      docTopicCounter.synchronized {
        val sv = dSparse(docTopicCounter)
        val d = (sv._1, generateAlias(sv._2, sv._1))
        cacheMap.update(docId, new SoftReference(d))
        d
      }
    }
  }

  private def wordTable(
    updateFunc: SoftReference[(Double, Table)] => Boolean,
    cacheMap: AppendOnlyMap[VertexId, SoftReference[(Double, Table)]],
    totalTopicCounter: BDV[Count],
    termTopicCounter: VD,
    termId: VertexId,
    numTerms: Double,
    beta: Double): (Double, Table) = {
    val cacheW = cacheMap(termId)
    if (!updateFunc(cacheW)) {
      cacheW.get
    } else {
      termTopicCounter.synchronized {
        val sv = wSparse(totalTopicCounter, termTopicCounter, numTerms, beta)
        val w = (sv._1, generateAlias(sv._2, sv._1))
        cacheMap.update(termId, new SoftReference(w))
        w
      }
    }
  }
}
