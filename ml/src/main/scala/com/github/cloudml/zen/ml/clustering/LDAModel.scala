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

import java.io._
import java.lang.ref.SoftReference
import java.util.Random

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, norm => brzNorm, sum => brzSum}
import com.github.cloudml.zen.ml.DBHPartitioner
import com.github.cloudml.zen.ml.clustering.LDAModel.{Count, DocId, ED, VD}
import com.github.cloudml.zen.ml.clustering.LDAUtils._
import com.github.cloudml.zen.ml.util.SparkUtils._
import com.github.cloudml.zen.ml.util.{AliasTable, LoaderUtils}
import com.google.common.base.Charsets
import com.google.common.io.Files
import org.apache.hadoop.io.{NullWritable, Text}
import org.apache.hadoop.mapred.TextOutputFormat
import org.apache.spark.graphx._
import org.apache.spark.graphx.impl.GraphImpl
import org.apache.spark.mllib.linalg.{DenseVector => SDV, SparseVector => SSV, Vector => SV}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.{Loader, MLUtils, Saveable}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.collection.AppendOnlyMap
import org.apache.spark.{Logging, SparkContext}
import org.json4s.DefaultFormats
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

class LocalLDAModel private[ml](
  private[ml] val gtc: BDV[Double],
  private[ml] val ttc: Array[BSV[Double]],
  val alpha: Double,
  val beta: Double,
  val alphaAS: Double) extends Serializable {

  def this(topicCounts: SDV, topicTermCounts: Array[SSV], alpha: Double, beta: Double) {
    this(new BDV[Double](topicCounts.toArray), topicTermCounts.map(t =>
      new BSV(t.indices, t.values, t.size)), alpha, beta, alpha)
  }

  @transient private lazy val numTopics = gtc.length
  @transient private lazy val numTerms = ttc.length
  @transient private lazy val numTokens = brzSum(gtc)
  @transient private lazy val betaSum = numTerms * beta
  @transient private lazy val alphaSum = numTopics * alpha
  @transient private lazy val termSum = numTokens + alphaAS * numTopics

  @transient private lazy val wordTableCache = new AppendOnlyMap[Int,
    SoftReference[(Double, AliasTable)]](ttc.length / 2)
  @transient private lazy val (t, tSum) = {
    val dv = LDAModel.tDense(gtc, numTokens, numTerms, alpha, alphaAS, beta)
    (AliasTable.generateAlias(dv._2, dv._1), dv._1)
  }
  @transient private lazy val rand = new Random()

  def setSeed(seed: Long): Unit = {
    rand.setSeed(seed)
  }

  def globalTopicCounter: SV = fromBreeze(gtc)

  def topicTermCounter: Array[SV] = ttc.map(t => fromBreeze(t))

  /**
   * inference interface
   * @param doc
   * @param totalIter overall iterations
   * @param burnIn
   */
  def inference(
    doc: SV,
    totalIter: Int = 10,
    burnIn: Int = 5): SV = {
    require(totalIter > burnIn, "totalIter is less than burnInIter")
    require(totalIter > 0, "totalIter is less than 0")
    require(burnIn > 0, "burnInIter is less than 0")

    val topicDist = BSV.zeros[Double](numTopics)
    val tokens = vector2Array(toBreeze(doc))
    val topics = new Array[Int](tokens.length)

    var docTopicCounter = uniformDistSampler(tokens, topics)
    for (i <- 1 to totalIter) {
      docTopicCounter = sampleTokens(docTopicCounter, tokens, topics)
      if (i > burnIn) topicDist :+= docTopicCounter
    }

    topicDist.compact()
    topicDist :/= brzNorm(topicDist, 1)
    fromBreeze(topicDist)
  }

  private[ml] def vector2Array(vec: BV[Double]): Array[Int] = {
    val docLen = brzSum(vec)
    var offset = 0
    val sent = new Array[Int](docLen.toInt)
    vec.activeIterator.filter(_._2 != 0.0).foreach { case (term, cn) =>
      for (i <- 0 until cn.toInt) {
        sent(offset) = term
        offset += 1
      }
    }
    sent
  }

  private[ml] def uniformDistSampler(
    tokens: Array[Int],
    topics: Array[Int]): BSV[Double] = {
    val docTopicCounter = BSV.zeros[Double](numTopics)
    for (i <- 0 until tokens.length) {
      val topic = uniformSampler(rand, numTopics)
      topics(i) = topic
      docTopicCounter(topic) += 1D
    }
    docTopicCounter
  }

  private[ml] def sampleTokens(
    docTopicCounter: BSV[Double],
    tokens: Array[Int],
    topics: Array[Int]): BSV[Double] = {
    for (i <- 0 until topics.length) {
      val termId = tokens(i)
      val currentTopic = topics(i)
      val d = LDAModel.dSparse(gtc, ttc(termId), docTopicCounter, currentTopic,
        numTokens, numTerms, alpha, alphaAS, beta)
      val (wSum, w) = wordTable(wordTableCache, gtc, ttc(termId), termId, numTokens, numTerms, alpha, alphaAS, beta)
      val newTopic = LDAModel.tokenSampling(rand, t, tSum, w, wSum, d)
      if (newTopic != currentTopic) {
        docTopicCounter(newTopic) += 1D
        docTopicCounter(currentTopic) -= 1D
        topics(i) = newTopic
        if (docTopicCounter(currentTopic) == 0) {
          docTopicCounter.compact()
        }
      }
    }
    docTopicCounter
  }

  private[ml] def wordTable(
    cacheMap: AppendOnlyMap[Int, SoftReference[(Double, AliasTable)]],
    totalTopicCounter: BDV[Double],
    termTopicCounter: BSV[Double],
    termId: Int,
    numTokens: Double,
    numTerms: Double,
    alpha: Double,
    alphaAS: Double,
    beta: Double): (Double, AliasTable) = {
    if (termTopicCounter.used == 0) return (0.0, null)
    var w = cacheMap(termId)
    if (w == null || w.get() == null) {
      val t = LDAModel.wSparse(totalTopicCounter, termTopicCounter, numTokens, numTerms, alpha, alphaAS, beta)
      w = new SoftReference((t._1, AliasTable.generateAlias(t._2, t._1)))
      cacheMap.update(termId, w)

    }
    w.get()
  }

  def save(path: String): Unit = {
    val file = new File(path)
    require(!file.exists, s"model file $path does exist")
    Files.touch(file)
    val fw = Files.newWriter(file, Charsets.UTF_8)
    fw.write(s"$numTopics $numTerms $alpha $beta $alphaAS \n")
    ttc.zipWithIndex.foreach { case (sv, index) =>
      val line = s"${index} ${sv.activeIterator.filter(_._2 != 0.0).map(t => s"${t._1}:${t._2}").mkString(" ")}\n"
      fw.write(line)
    }
    fw.flush()
    fw.close()
  }

}

class DistributedLDAModel private[ml](
  private[ml] val gtc: BDV[Count],
  private[ml] val ttc: RDD[(VertexId, VD)],
  val numTopics: Int,
  val numTerms: Long,
  val alpha: Double,
  val beta: Double,
  val alphaAS: Double) extends Serializable with Saveable with Logging {

  @transient private lazy val numTokens = brzSum(gtc)
  @transient private lazy val betaSum = numTerms * beta
  @transient private lazy val alphaSum = numTopics * alpha
  @transient private lazy val termSum = numTokens + alphaAS * numTopics
  @transient private var seed = new Random().nextInt()
  @transient private var innerIter = 1
  val storageLevel: StorageLevel = ttc.getStorageLevel

  /**
   * inference interface
   * @param docs tuple pair: (dicId, Vector), in which 'docId' is unique
   *             recommended storage level: StorageLevel.MEMORY_AND_DISK
   * @param totalIter overall iterations
   * @param burnIn previous burnIn iters results will discard
   */
  def inference(
    docs: RDD[(VertexId, SV)],
    totalIter: Int = 25,
    burnIn: Int = 22): RDD[(VertexId, SV)] = {
    require(totalIter > burnIn, "totalIter is less than burnInIter")
    require(totalIter > 0, "totalIter is less than 0")
    require(burnIn > 0, "burnIn is less than 0")
    var corpus: Graph[VD, ED] = initializeInferDataset(docs, numTopics, storageLevel)
    var docTopicCounter: RDD[(VertexId, VD)] = null

    for (iter <- 1 to totalIter) {
      logInfo(s"Start Gibbs sampling (Iteration $iter/$totalIter)")
      val startedAt = System.nanoTime()
      corpus = gibbsSampling(corpus, iter)
      val elapsedSeconds = (System.nanoTime() - startedAt) / 1e9
      logInfo(s"End Gibbs sampling  (Iteration $iter/$totalIter) takes:  $elapsedSeconds")

      if (iter > burnIn) {
        var previousDocTopicCounter = docTopicCounter
        val newTermTopicCounter = corpus.vertices.filter(t => t._1 < 0)
        docTopicCounter = Option(docTopicCounter).map(_.join(newTermTopicCounter).map {
          case (docId, (a, b)) =>
            (docId, a :+ b)
        }).getOrElse(newTermTopicCounter)

        docTopicCounter.persist(storageLevel).count()
        Option(previousDocTopicCounter).foreach(_.unpersist(blocking = false))
        previousDocTopicCounter = docTopicCounter
      }
    }
    docTopicCounter.map { case (docId, sv) =>
      sv.compact()
      sv :/= brzNorm(sv, 1)
      (toDocId(docId), fromBreeze(sv))
    }
  }

  def toLocalLDAModel(): LocalLDAModel = {
    val ttc1 = Array.fill(numTerms.toInt) {
      BSV.zeros[Double](numTopics)
    }
    ttc.collect().foreach { case (termId, vector) =>
      ttc1(termId.toInt) :+= vector
    }
    new LocalLDAModel(gtc, ttc1, alpha, beta, alphaAS)
  }

  private[ml] def initializeInferDataset(docs: RDD[(LDA.DocId, SV)],
    numTopics: Int,
    storageLevel: StorageLevel): Graph[VD, ED] = {
    val previousCorpus: Graph[VD, ED] = initializeCorpus(docs, numTopics, storageLevel)
    val corpus = previousCorpus.outerJoinVertices(ttc) { (vid, c, v) =>
      if (vid >= 0) {
        assert(v.isDefined)
      }
      v.getOrElse(c)
    }
    corpus.persist(storageLevel)
    corpus.vertices.count()
    corpus.edges.count()
    previousCorpus.edges.unpersist(blocking = false)
    previousCorpus.vertices.unpersist(blocking = false)
    corpus
  }

  private[ml] def initializeCorpus(
    docs: RDD[(LDA.DocId, SV)],
    numTopics: Int,
    storageLevel: StorageLevel): Graph[VD, ED] = {
    val edges = docs.mapPartitionsWithIndex((pid, iter) => {
      val gen = new Random(pid + 117)
      iter.flatMap { case (docId, doc) =>
        initializeEdges(gen, doc, docId, numTopics)
      }
    })
    edges.persist(storageLevel)
    var corpus: Graph[VD, ED] = Graph.fromEdges(edges, null, storageLevel, storageLevel)
    corpus.persist(storageLevel)
    corpus.vertices.count()
    corpus.edges.count()
    edges.unpersist(blocking = false)
    corpus = DBHPartitioner.partitionByDBH[VD, ED](corpus, storageLevel)
    updateCounter(corpus, numTopics)
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

  private def genNewDocId(docId: Long): Long = {
    -(docId + 1L)
  }

  private def toDocId(docId: Long): Long = {
    -1L - docId
  }

  private def gibbsSampling(_corpus: Graph[VD, ED], iter: Int): Graph[VD, ED] = {
    var corpus = _corpus
    val totalTopicCounter = gtc
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

    previousCorpus.edges.unpersist(blocking = false)
    previousCorpus.vertices.unpersist(blocking = false)
    sampledCorpus.edges.unpersist(blocking = false)
    sampledCorpus.vertices.unpersist(blocking = false)
    innerIter += 1
    corpus
  }

  private def checkpoint(corpus: Graph[VD, ED]): Unit = {
    val sc = corpus.edges.sparkContext
    if (innerIter % 10 == 0 && sc.getCheckpointDir.isDefined) {
      corpus.checkpoint()
    }
  }

  private def sampleTokens(
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
        // table is a per term data structure
        // in GraphX, edges in a partition are clustered by source IDs (term id in this case)
        // so, use below simple cache to avoid calculating table each time
        val lastWTable = new AliasTable(numTopics.toInt)
        var lastVid: VertexId = -1
        var lastWSum = 0.0
        val dv = LDAModel.tDense(totalTopicCounter, numTokens, numTerms, alpha, alphaAS, beta)
        val t = AliasTable.generateAlias(dv._2, dv._1)
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
              val d = docTopicCounter.synchronized {
                LDAModel.dSparse(gtc, termTopicCounter, docTopicCounter, currentTopic,
                  numTokens, numTerms, alpha, alphaAS, beta)
              }

              if (lastVid != termId) {
                lastWSum = wordTable(lastWTable, totalTopicCounter, termTopicCounter,
                  termId, numTokens, numTerms, alpha, alphaAS, beta)
                lastVid = termId
              }
              val newTopic = LDAModel.tokenSampling(gen, t, tSum, lastWTable, lastWSum, d)

              if (newTopic != currentTopic) {
                topics(i) = newTopic
                docTopicCounter.synchronized {
                  val cn = docTopicCounter(currentTopic)
                  docTopicCounter(currentTopic) = cn - 1D
                  docTopicCounter(newTopic) += 1D
                  // if (cn == 1D) docTopicCounter.compact()
                }
              }
            }

            topics
        }
      }, TripletFields.All)

    GraphImpl(nweGraph.vertices.mapValues { (vid, cn) =>
      if (vid < 0) null.asInstanceOf[VD] else cn
    }, nweGraph.edges)
  }

  private def wordTable(
    table: AliasTable,
    totalTopicCounter: BDV[Count],
    termTopicCounter: VD,
    termId: VertexId,
    numTokens: Double,
    numTerms: Double,
    alpha: Double,
    alphaAS: Double,
    beta: Double): Double = {
    val sv = LDAModel.wSparse(totalTopicCounter, termTopicCounter,
      numTokens, numTerms, alpha, alphaAS, beta)
    AliasTable.generateAlias(sv._2, sv._1, table)
    sv._1
  }

  private def updateCounter(graph: Graph[VD, ED], numTopics: Int): Graph[VD, ED] = {
    val newCounter = graph.aggregateMessages[VD](ctx => {
      val topics = ctx.attr
      val vector = BSV.zeros[Count](numTopics)
      for (topic <- topics) {
        vector(topic) += 1
      }
      ctx.sendToDst(vector)
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
    graph.joinVertices(newCounter) { case (_, o, n) => n }
  }

  override def save(sc: SparkContext, path: String): Unit = {
    save(sc, path, isTransposed = false)
  }

  def save(path: String): Unit = {
    save(ttc.context, path, isTransposed = false)
  }

  /**
   * @param sc
   * @param path
   * @param isTransposed libsvm when `isTransposed` is false, the format of each line:
   *                     termId  \grave{topicId}:counter \grave{topicId}:counter...,
   *                     in which \grave{topicId} = topicId + 1
   *                     otherwise:
   *                     topicId \grave{termId}:counter \grave{termId}:counter...,
   *                     in which \grave{termId}= termId + 1
   */
  def save(sc: SparkContext, path: String, isTransposed: Boolean): Unit = {
    LDAModel.SaveLoadV1_0.save(sc, path, ttc, numTopics, numTerms, alpha, beta, alphaAS, isTransposed)
  }

  override protected def formatVersion: String = LDAModel.SaveLoadV1_0.formatVersionV1_0
}

object LDAModel extends Loader[DistributedLDAModel] {
  private[ml] type DocId = VertexId
  private[ml] type WordId = VertexId
  private[ml] type Count = Double
  private[ml] type ED = Array[Int]
  private[ml] type VD = BSV[Count]

  private[ml] def tokenSampling(
    gen: Random,
    t: AliasTable,
    tSum: Double,
    w: AliasTable,
    wSum: Double,
    d: BSV[Double]): Int = {
    val index = d.index
    val data = d.data
    val used = d.used
    val dSum = data(d.used - 1)
    val distSum = tSum + wSum + dSum
    val genSum = gen.nextDouble() * distSum
    if (genSum < dSum) {
      val dGenSum = gen.nextDouble() * dSum
      val pos = binarySearchInterval(data, dGenSum, 0, used, greater = true)
      index(pos)
    } else if (genSum < (dSum + wSum)) {
      w.sampleAlias(gen)
    } else {
      t.sampleAlias(gen)
    }
  }

  private[ml] def wSparse(
    totalTopicCounter: BDV[Double],
    termTopicCounter: BSV[Double],
    numTokens: Double,
    numTerms: Double,
    alpha: Double,
    alphaAS: Double,
    beta: Double): (Double, BSV[Double]) = {
    val numTopics = totalTopicCounter.length
    val betaSum = numTerms * beta
    val alphaSum = numTopics * alpha
    val termSum = numTokens + alphaAS * numTopics

    val w = BSV.zeros[Double](numTopics)
    var sum = 0.0
    termTopicCounter.activeIterator.foreach { t =>
      val topic = t._1
      val count = t._2
      val last = count * alphaSum * (totalTopicCounter(topic) + alphaAS) /
        ((totalTopicCounter(topic) + betaSum) * termSum)
      w(topic) = last
      sum += last
    }
    (sum, w)
  }

  private[ml] def tDense(
    totalTopicCounter: BDV[Double],
    numTokens: Double,
    numTerms: Double,
    alpha: Double,
    alphaAS: Double,
    beta: Double): (Double, BDV[Double]) = {
    val numTopics = totalTopicCounter.length
    val betaSum = numTerms * beta
    val alphaSum = numTopics * alpha
    val termSum = numTokens + alphaAS * numTopics
    val t = BDV.zeros[Double](numTopics)
    var sum = 0.0
    for (topic <- 0 until numTopics) {
      val last = beta * alphaSum * (totalTopicCounter(topic) + alphaAS) /
        ((totalTopicCounter(topic) + betaSum) * termSum)
      t(topic) = last
      sum += last
    }
    (sum, t)
  }

  private[ml] def dSparse(
    totalTopicCounter: BDV[Double],
    termTopicCounter: BSV[Double],
    docTopicCounter: BSV[Double],
    currentTopic: Int,
    numTokens: Double,
    numTerms: Double,
    alpha: Double,
    alphaAS: Double,
    beta: Double): BSV[Double] = {
    val numTopics = totalTopicCounter.length
    // val termSum = numTokens - 1D + alphaAS * numTopics
    val betaSum = numTerms * beta
    val d = BSV.zeros[Double](numTopics)
    var sum = 0.0
    docTopicCounter.activeIterator.filter(_._2 > 0D).foreach { t =>
      val topic = t._1
      val count = if (currentTopic == topic && t._2 != 1) t._2 - 1 else t._2
      // val last = count * termSum * (termTopicCounter(topic) + beta) /
      //  ((totalTopicCounter(topic) + betaSum) * termSum)
      val last = count * (termTopicCounter(topic) + beta) /
        (totalTopicCounter(topic) + betaSum)
      sum += last
      d(topic) = sum
    }
    d
  }

  override def load(sc: SparkContext, path: String): DistributedLDAModel = {
    val (loadedClassName, version, metadata) = LoaderUtils.loadMetadata(sc, path)
    val versionV1_0 = SaveLoadV1_0.formatVersionV1_0
    val classNameV1_0 = SaveLoadV1_0.classNameV1_0
    if (loadedClassName == classNameV1_0 && version == versionV1_0) {
      implicit val formats = DefaultFormats
      val alpha = (metadata \ "alpha").extract[Double]
      val beta = (metadata \ "beta").extract[Double]
      val alphaAS = (metadata \ "alphaAS").extract[Double]
      val numTopics = (metadata \ "numTopics").extract[Int]
      val numTerms = (metadata \ "numTerms").extract[Long]
      val isTransposed = (metadata \ "isTransposed").extract[Boolean]
      val rdd = SaveLoadV1_0.loadData(sc, path, isTransposed, classNameV1_0, numTopics, numTerms)

      val ttc = if (isTransposed) {
        rdd.flatMap { case (topicId, vector) =>
          vector.activeIterator.map { case (termId, cn) =>
            val z = BSV.zeros[Double](numTopics)
            z(topicId.toInt) = cn
            (termId.toLong, z)
          }
        }.reduceByKey(_ :+ _).map {
          t => t._2.compact(); t
        }
      } else {
        rdd
      }
      ttc.persist(StorageLevel.MEMORY_AND_DISK)

      val gtc = ttc.map(_._2).aggregate(BDV.zeros[Count](numTopics))(_ :+= _, _ :+= _)
      new DistributedLDAModel(gtc, ttc, numTopics, numTerms, alpha, beta, alphaAS)
    } else {
      throw new Exception(
        s"LDAModel.load did not recognize model with (className, format version):" +
          s"($loadedClassName, $version).  Supported:\n" +
          s"  ($classNameV1_0, 1.0)")
    }

  }

  def loadLocalLDAModel(filePath: String): LocalLDAModel = {
    val file: File = new File(filePath)
    require(file.exists, s"model file $filePath does not exist")
    require(file.isFile, s"model file $filePath is not a normal file")
    val lines = Files.readLines(file, Charsets.UTF_8)
    val Array(numTopics, numTerms, alpha, beta, alphaAS) = lines.get(0).split(" ")
    val ttc = Array.fill(numTerms.toInt) {
      BSV.zeros[Double](numTopics.toInt)
    }
    val iter = lines.listIterator(1)
    while (iter.hasNext) {
      val line = iter.next.trim
      if (!line.isEmpty && !line.startsWith("#")) {
        val its = line.split(" ")
        val offset = its.head.toInt
        val sv = ttc(offset)
        its.tail.foreach { s =>
          val Array(index, value) = s.split(":")
          sv(index.toInt) = value.toDouble
        }
        sv.compact()

      }
    }
    val gtc = BDV.zeros[Double](numTopics.toInt)
    ttc.foreach { tc =>
      gtc :+= tc
    }
    new LocalLDAModel(gtc, ttc, alpha.toDouble, beta.toDouble, alphaAS.toDouble)
  }

  private[ml] object SaveLoadV1_0 {

    val formatVersionV1_0 = "1.0"
    val classNameV1_0 = "com.github.cloudml.zen.ml.clustering.DistributedLDAModel"

    def loadData(
      sc: SparkContext,
      path: String,
      isTransposed: Boolean,
      modelClass: String,
      numTopics: Int,
      numTerms: Long): RDD[(VertexId, VD)] = {
      val dataPath = LoaderUtils.dataPath(path)
      val numSize = if (isTransposed) numTerms.toInt else numTopics
      sc.textFile(dataPath).map { line =>
        val sv = BSV.zeros[Double](numSize)
        val arr = line.split("\t")
        arr.tail.foreach { sub =>
          val Array(index, value) = sub.split(":")
          sv(index.toInt) = value.toDouble
        }
        sv.compact()
        (arr.head.toLong, sv)
      }
    }

    def loadDataFromSolidFile(sc: SparkContext,
                              path: String): DistributedLDAModel = {
      type MT = Tuple6[Int, Long, Double, Double, Double, Boolean]
      val (metas, rdd) = LoaderUtils.HDFSFile2RDD[(VertexId, VD), MT](sc, path, header => {
        implicit val formats = DefaultFormats
        val metadata = parse(header)
        val alpha = (metadata \ "alpha").extract[Double]
        val beta = (metadata \ "beta").extract[Double]
        val alphaAS = (metadata \ "alphaAS").extract[Double]
        val numTopics = (metadata \ "numTopics").extract[Int]
        val numTerms = (metadata \ "numTerms").extract[Long]
        val isTransposed = (metadata \ "isTransposed").extract[Boolean]
        (numTopics, numTerms, alpha, beta, alphaAS, isTransposed)
      }, (metas, line) => {
        val numTopics = metas._1
        val numTerms = metas._2
        val isTransposed = metas._6
        val numSize = if (isTransposed) numTerms.toInt else numTopics
        val sv = BSV.zeros[Double](numSize)
        val arr = line.split("\t")
        arr.tail.foreach { sub =>
          val Array(index, value) = sub.split(":")
          sv(index.toInt) = value.toDouble
        }
        sv.compact()
        (arr.head.toLong, sv)
      })

      val (numTopics, numTerms, alpha, beta, alphaAS, isTransposed) = metas
      val ttc = if (isTransposed) {
        rdd.flatMap { case (topicId, vector) =>
          vector.activeIterator.map { case (termId, cn) =>
            val z = BSV.zeros[Double](numTopics)
            z(topicId.toInt) = cn
            (termId.toLong, z)
          }
        }.reduceByKey(_ += _).map {
          t => t._2.compact(); t
        }
      } else {
        rdd
      }
      ttc.persist(StorageLevel.MEMORY_AND_DISK)

      val gtc = ttc.map(_._2).aggregate(BDV.zeros[Count](numTopics))(_ :+= _, _ :+= _)

      new DistributedLDAModel(gtc, ttc, numTopics, numTerms, alpha, beta, alphaAS)
    }

    def save(
      sc: SparkContext,
      path: String,
      ttc: RDD[(VertexId, VD)],
      numTopics: Int,
      numTerms: Long,
      alpha: Double,
      beta: Double,
      alphaAS: Double,
      isTransposed: Boolean,
      saveSolid: Boolean = true): Unit = {
      val metadata = compact(render
        (("class" -> classNameV1_0) ~ ("version" -> formatVersionV1_0) ~
          ("alpha" -> alpha) ~ ("beta" -> beta) ~ ("alphaAS" -> alphaAS) ~
          ("numTopics" -> numTopics) ~ ("numTerms" -> numTerms) ~
          ("numEdges" -> LDA.numEdges) ~ ("numDocs" -> LDA.numDocs)
          ~ ("isTransposed" -> isTransposed)))

      val rdd = if (isTransposed) {
        ttc.flatMap { case (termId, vector) =>
          vector.activeIterator.map { case (topicId, cn) =>
            val z = BSV.zeros[Double](numTerms.toInt)
            z(termId.toInt) = cn
            (topicId.toLong, z)
          }
        }.reduceByKey(_ :+ _)
      } else {
        ttc
      }

      if (saveSolid) {
        val metadata_line = metadata.replaceAll("\n", "")
        val rdd_txt = rdd.map { case (id, vector) =>
          val list = vector.activeIterator.toList.sortWith((a, b) => a._2 > b._2)
          id.toString + "\t" + list.map(item => item._1 + ":" + item._2).mkString("\t")
        }
        LoaderUtils.RDD2HDFSFile[String](sc, rdd_txt, path, metadata_line, t => t)
      } else {
        sc.parallelize(Seq(metadata), 1).saveAsTextFile(LoaderUtils.metadataPath(path))
        // save model with the topic or word-term descending order
        rdd.map { case (id, vector) =>
          val list = vector.activeIterator.toList.sortWith((a, b) => a._2 > b._2)
          (NullWritable.get(), new Text(id + "\t" + list.map(item => item._1 + ":" + item._2).mkString("\t")))
        }.saveAsHadoopFile[TextOutputFormat[NullWritable, Text]](LoaderUtils.dataPath(path))
      }
    }
  }

}

private[ml] object LDAUtils {

  def uniformSampler(rand: Random, dimension: Int): Int = {
    rand.nextInt(dimension)
  }

  def binarySearchInterval(
    index: Array[Double],
    key: Double,
    begin: Int,
    end: Int,
    greater: Boolean): Int = {
    if (begin == end) {
      return if (greater) end else begin - 1
    }
    var b = begin
    var e = end - 1

    var mid: Int = (e + b) >> 1
    while (b <= e) {
      mid = (e + b) >> 1
      val v = index(mid)
      if (v < key) {
        b = mid + 1
      }
      else if (v > key) {
        e = mid - 1
      }
      else {
        return mid
      }
    }
    val v = index(mid)
    mid = if ((greater && v >= key) || (!greater && v <= key)) {
      mid
    }
    else if (greater) {
      mid + 1
    }
    else {
      mid - 1
    }

    if (greater) {
      if (mid < end) assert(index(mid) >= key)
      if (mid > 0) assert(index(mid - 1) <= key)
    } else {
      if (mid > 0) assert(index(mid) <= key)
      if (mid < end - 1) assert(index(mid + 1) >= key)
    }
    mid
  }
}
