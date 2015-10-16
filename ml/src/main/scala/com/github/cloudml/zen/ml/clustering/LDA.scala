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

import java.util.Random
import java.util.concurrent.Executors
import scala.concurrent._
import scala.concurrent.duration._

import LDADefines._
import com.github.cloudml.zen.ml.partitioner._
import com.github.cloudml.zen.ml.util.{SparkUtils, XORShiftRandom}

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV}
import org.apache.hadoop.fs.Path
import org.apache.spark.graphx2._
import org.apache.spark.graphx2.impl.GraphImpl
import org.apache.spark.mllib.linalg.{SparseVector => SSV, Vector => SV}
import org.apache.spark.mllib.linalg.distributed.{MatrixEntry, RowMatrix}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel


class LDA(@transient var corpus: Graph[TC, TA],
  val numTopics: Int,
  val numTerms: Int,
  val numDocs: Long,
  val numTokens: Long,
  var alpha: Double,
  var beta: Double,
  var alphaAS: Double,
  val algo: LDAAlgorithm,
  var storageLevel: StorageLevel) extends Serializable {

  @transient var topicCounters: BDV[Count] = null
  @transient lazy val seed = new XORShiftRandom().nextInt()

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

  def getCorpus: Graph[TC, TA] = corpus

  def termVertices: VertexRDD[TC] = corpus.vertices.filter(t => isTermId(t._1))

  def docVertices: VertexRDD[TC] = corpus.vertices.filter(t => isDocId(t._1))

  private def scConf = corpus.edges.context.getConf

  def init(computedModel: Option[RDD[(VertexId, TC)]] = None): Unit = {
    corpus = algo.updateVertexCounters(corpus, numTopics)
    corpus = computedModel match {
      case Some(cm) =>
        val verts = corpus.vertices.leftJoin(cm)((_, uc, cc) => cc.getOrElse(uc))
        GraphImpl.fromExistingRDDs(verts, corpus.edges)
      case None => corpus
    }
    corpus.vertices.persist(storageLevel).setName("vertices-0")
    collectTopicCounters()
  }

  def collectTopicCounters(): Unit = {
    val numTopics = this.numTopics
    val numThreads = scConf.getInt(cs_numThreads, 1)
    val graph = corpus.asInstanceOf[GraphImpl[TC, TA]]
    val aggRdd = graph.vertices.partitionsRDD.mapPartitions(_.map(svp => {
      val totalSize = svp.capacity
      val index = svp.index
      val mask = svp.mask
      val values = svp.values
      val sizePerthrd = {
        val npt = totalSize / numThreads
        if (npt * numThreads == totalSize) npt else npt + 1
      }
      implicit val es = ExecutionContext.fromExecutorService(Executors.newFixedThreadPool(numThreads))
      val all = Range(0, numThreads).map(thid => Future {
        val startPos = sizePerthrd * thid
        val endPos = math.min(sizePerthrd * (thid + 1), totalSize)
        val agg = BDV.zeros[Count](numTopics)
        var pos = mask.nextSetBit(startPos)
        while (pos < endPos && pos >= 0) {
          if (isTermId(index.getValue(pos))) values(pos) match {
            case u: BDV[Count] => agg :+= u
            case u: BSV[Count] => agg :+= u
            case _ =>
          }
          pos = mask.nextSetBit(pos + 1)
        }
        agg
      })
      val aggs = Await.result(Future.sequence(all), 1.hour)
      es.shutdown()
      aggs.reduce(_ :+= _)
    }))
    val gtc = aggRdd.collect().par.reduce(_ :+= _)
    val count = gtc.data.par.map(_.toLong).sum
    assert(count == numTokens, s"numTokens=$numTokens, count=$count")
    topicCounters = gtc
  }

  def runGibbsSampling(totalIter: Int): Unit = {
    val pplx = scConf.getBoolean(cs_calcPerplexity, false)
    val saveIntv = scConf.getInt(cs_saveInterval, 0)
    if (pplx) {
      println("Before Gibbs sampling:")
      LDAPerplexity(this).output(println)
    }
    var iter = 1
    while (iter <= totalIter) {
      println(s"\nStart Gibbs sampling (Iteration $iter/$totalIter)")
      val startedAt = System.nanoTime()
      gibbsSampling(iter)
      if (pplx) {
        LDAPerplexity(this).output(println)
      }
      if (saveIntv > 0 && iter % saveIntv == 0) {
        val sc = corpus.edges.context
        val model = toLDAModel()
        val savPath = new Path(scConf.get(cs_outputpath) + s"-iter$iter")
        val fs = SparkUtils.getFileSystem(scConf, savPath)
        fs.delete(savPath, true)
        model.save(sc, savPath.toString, isTransposed=true)
        println(s"Model saved after Iteration $iter")
      }
      val elapsedSeconds = (System.nanoTime() - startedAt) / 1e9
      println(s"End Gibbs sampling (Iteration $iter/$totalIter) takes total: $elapsedSeconds secs")
      iter += 1
    }
  }

  def gibbsSampling(sampIter: Int, inferenceOnly: Boolean = false): Unit = {
    val chkptIntv = scConf.getInt(cs_chkptInterval, 0)
    val prevCorpus = corpus
    val sampledCorpus = algo.sampleGraph(corpus, topicCounters, seed, sampIter,
      numTokens, numTopics, numTerms, alpha, alphaAS, beta)
    corpus = algo.updateVertexCounters(sampledCorpus, numTopics, inferenceOnly)
    if (chkptIntv > 0 && sampIter % chkptIntv == 1) {
      if (corpus.edges.context.getCheckpointDir.isDefined) {
        corpus.checkpoint()
      }
    }
    corpus.persist(storageLevel)
    val startedAt = System.nanoTime()
    corpus.edges.setName(s"edges-$sampIter").count()
    corpus.vertices.setName(s"vertices-$sampIter")
    collectTopicCounters()
    val elapsedSeconds = (System.nanoTime() - startedAt) / 1e9
    println(s"Sampling & update paras $sampIter takes: $elapsedSeconds secs")
    prevCorpus.unpersist(blocking=false)
  }

  /**
   * run more iters, return averaged counters
   * @param filter return which vertices
   * @param runIter saved more these iters' averaged model
   */
  def runSum(filter: VertexId => Boolean,
    runIter: Int = 0,
    inferenceOnly: Boolean = false): RDD[(VertexId, TC)] = {
    def vertices = corpus.vertices.filter(t => filter(t._1))
    var countersSum = vertices
    countersSum.persist(storageLevel)
    var iter = 1
    while (iter <= runIter) {
      println(s"Save TopicModel (Iteration $iter/$runIter)")
      gibbsSampling(iter, inferenceOnly)
      countersSum = countersSum.innerZipJoin(vertices)((_, a, b) => a :+= b)
      countersSum.persist(storageLevel)
      iter += 1
    }
    countersSum
  }

  def toLDAModel(runIter: Int = 0): DistributedLDAModel = {
    val ttcsSum = runSum(isTermId, runIter)
    val ttcs = if (runIter == 0) {
      ttcsSum
    } else {
      val turn = (runIter + 1).toDouble
      ttcsSum.mapValues(_.mapValues(v => {
        val gen = new XORShiftRandom()
        val aver = v / turn
        val intPart = math.floor(aver)
        if (gen.nextDouble() > aver - intPart) intPart else intPart + 1
      }.toInt))
    }
    ttcs.persist(storageLevel)
    new DistributedLDAModel(ttcs, numTopics, numTerms, numTokens, alpha, beta, alphaAS, storageLevel)
  }

  def mergeDuplicateTopic(threshold: Double = 0.95D): Map[Int, Int] = {
    val rows = termVertices.map(t => t._2).map(v => {
      val length = v.length
      val index = v.activeKeysIterator.toArray
      val data = v.activeValuesIterator.toArray.map(_.toDouble)
      new SSV(length, index, data).asInstanceOf[SV]
    })
    val simMatrix = new RowMatrix(rows).columnSimilarities()
    val minMap = simMatrix.entries.filter {
      case MatrixEntry(row, column, sim) => sim > threshold && row != column
    }.map {
      case MatrixEntry(row, column, sim) => (column.toInt, row.toInt)
    }.groupByKey().map {
      case (topic, simTopics) => (topic, simTopics.min)
    }.collect().toMap
    if (minMap.nonEmpty) {
      val mergingCorpus = corpus.mapEdges(_.attr.map(topic =>
        minMap.getOrElse(topic, topic))
      )
      corpus = algo.updateVertexCounters(mergingCorpus, numTopics)
    }
    minMap
  }
}

object LDA {
  def apply(docs: EdgeRDD[TA],
    numTopics: Int,
    alpha: Double,
    beta: Double,
    alphaAS: Double,
    algo: LDAAlgorithm,
    storageLevel: StorageLevel): LDA = {
    val initCorpus: Graph[TC, TA] = LBVertexRDDBuilder.fromEdgeRDD(docs, storageLevel)
    initCorpus.persist(storageLevel)
    val numTokens = initCorpus.edges.map(_.attr.length.toLong).reduce(_ + _)
    println(s"tokens in the corpus: $numTokens")
    val vertices = initCorpus.vertices
    val numTerms = vertices.map(_._1).filter(isTermId).count().toInt
    println(s"terms in the corpus: $numTerms")
    val numDocs = vertices.map(_._1).filter(isDocId).count()
    println(s"docs in the corpus: $numDocs")
    val lda = new LDA(initCorpus, numTopics, numTerms, numDocs, numTokens, alpha, beta, alphaAS,
      algo, storageLevel)
    lda.init()
    vertices.unpersist(blocking=false)
    lda
  }

  // initialize LDA for inference or incremental training
  def apply(computedModel: DistributedLDAModel,
    docs: EdgeRDD[TA],
    algo: LDAAlgorithm): LDA = {
    val numTopics = computedModel.numTopics
    val numTerms = computedModel.numTerms
    val numTokens = computedModel.numTokens
    val alpha = computedModel.alpha
    val beta = computedModel.beta
    val alphaAS = computedModel.alphaAS
    val storageLevel = computedModel.storageLevel
    println(s"tokens in the corpus: $numTokens")
    println(s"terms in the corpus: $numTerms")
    val initCorpus: Graph[TC, TA] = LBVertexRDDBuilder.fromEdgeRDD(docs, storageLevel)
    initCorpus.persist(storageLevel)
    val vertices = initCorpus.vertices
    val numDocs = vertices.map(_._1).filter(isDocId).count()
    println(s"docs in the corpus: $numDocs")
    val lda = new LDA(initCorpus, numTopics, numTerms, numDocs, numTokens, alpha, beta, alphaAS,
      algo, storageLevel)
    lda.init(Some(computedModel.termTopicsRDD))
    vertices.unpersist(blocking=false)
    lda
  }

  /**
   * LDA training
   * @param docs       EdgeRDD of corpus
   * @param totalIter  the number of iterations
   * @param numTopics  the number of topics (5000+ for large data)
   * @param alpha      recommend to be (5.0 /numTopics)
   * @param beta       recommend to be in range 0.001 - 0.1
   * @param alphaAS    recommend to be in range 0.01 - 1.0
   * @param storageLevel StorageLevel that the LDA Model RDD uses
   * @return DistributedLDAModel
   */
  def train(docs: EdgeRDD[TA],
    totalIter: Int,
    numTopics: Int,
    alpha: Double,
    beta: Double,
    alphaAS: Double,
    storageLevel: StorageLevel): DistributedLDAModel = {
    val conf = docs.context.getConf
    val algo: LDAAlgorithm = conf.get(cs_LDAAlgorithm, "fastlda") match {
      case "lightlda" =>
        println("using LightLDA sampling algorithm.")
        new LightLDA
      case "fastlda" =>
        println("using FastLDA sampling algorithm.")
        new FastLDA
      case "sparselda" =>
        println("using SparseLDA sampling algorithm")
        new SparseLDA
      case _ =>
        throw new NoSuchMethodException("No this algorithm or not implemented.")
    }
    val lda = LDA(docs, numTopics, alpha, beta, alphaAS, algo, storageLevel)
    lda.runGibbsSampling(totalIter)
    lda.toLDAModel()
  }

  def incrementalTrain(docs: EdgeRDD[TA],
    computedModel: DistributedLDAModel,
    totalIter: Int,
    storageLevel: StorageLevel): DistributedLDAModel = {
    val conf = docs.context.getConf
    val algo: LDAAlgorithm = conf.get(cs_LDAAlgorithm, "fastlda") match {
      case "lightlda" =>
        println("using LightLDA sampling algorithm.")
        new LightLDA
      case "fastlda" =>
        println("using FastLDA sampling algorithm.")
        new FastLDA
      case "sparselda" =>
        println("using SparseLDA sampling algorithm")
        new SparseLDA
      case _ =>
        throw new NoSuchMethodException("No this algorithm or not implemented.")
    }
    val lda = LDA(computedModel, docs, algo)
    var iter = 1
    while (iter <= 15) {
      lda.gibbsSampling(iter, inferenceOnly=true)
      iter += 1
    }
    lda.runGibbsSampling(totalIter)
    lda.toLDAModel()
  }

  /**
   * @param orgDocs  RDD of documents, which are term (word) count vectors paired with IDs.
   *                 The term count vectors are "bags of words" with a fixed-size vocabulary
   *                 (where the vocabulary size is the length of the vector).
   *                 Document IDs must be unique and >= 0.
   */
  def initializeCorpusEdges(orgDocs: RDD[_],
    docType: String,
    numTopics: Int,
    reverse: Boolean,
    storageLevel: StorageLevel): EdgeRDD[TA] = {
    val conf = orgDocs.context.getConf
    val ignDid = conf.getBoolean(cs_ignoreDocId, false)
    val docs = docType match {
      case "raw" => convertRawDocs(orgDocs.asInstanceOf[RDD[String]], numTopics, ignDid, reverse)
      case "bow" => convertBowDocs(orgDocs.asInstanceOf[RDD[BOW]], numTopics, ignDid, reverse)
    }
    val initCorpus: Graph[TC, TA] = LBVertexRDDBuilder.fromEdges(docs, storageLevel)
    initCorpus.persist(storageLevel)
    val partCorpus = conf.get(cs_partStrategy, "dbh") match {
      case "byterm" =>
        println("partition corpus by terms.")
        if (reverse) {
          EdgeDstPartitioner.partitionByEDP[TC, TA](initCorpus, storageLevel)
        } else {
          initCorpus.partitionBy(PartitionStrategy.EdgePartition1D)
        }
      case "bydoc" =>
        println("partition corpus by docs.")
        if (reverse) {
          initCorpus.partitionBy(PartitionStrategy.EdgePartition1D)
        } else {
          EdgeDstPartitioner.partitionByEDP[TC, TA](initCorpus, storageLevel)
        }
      case "edge2d" =>
        println("using Edge2D partition strategy.")
        initCorpus.partitionBy(PartitionStrategy.EdgePartition2D)
      case "dbh" =>
        println("using Degree-based Hashing partition strategy.")
        DBHPartitioner.partitionByDBH[TC, TA](initCorpus, storageLevel)
      case "vsdlp" =>
        println("using Vertex-cut Stochastic Dynamic Label Propagation partition strategy.")
        VSDLPPartitioner.partitionByVSDLP[TC, TA](initCorpus, 4, storageLevel)
      case "bbr" =>
        println("using Bounded & Balanced Rearranger partition strategy.")
        BBRPartitioner.partitionByBBR[TC, TA](initCorpus, storageLevel)
      case _ =>
        throw new NoSuchMethodException("No this algorithm or not implemented.")
    }
    val reCorpus = resampleCorpus(partCorpus, numTopics, storageLevel)
    val edges = reCorpus.edges
    val numEdges = edges.persist(storageLevel).setName("edges-0").count()
    println(s"edges in the corpus: $numEdges")
    initCorpus.unpersist(blocking=false)
    edges
  }

  def convertRawDocs(rawDocs: RDD[String],
    numTopics: Int,
    ignDid: Boolean,
    reverse: Boolean): RDD[Edge[TA]] = {
    rawDocs.mapPartitionsWithIndex((pid, iter) => {
      val gen = new XORShiftRandom(pid + 117)
      var pidMark = pid.toLong << 48
      iter.flatMap(line => {
        val tokens = line.split(raw"\t|\s+")
        val docId = if (ignDid) {
          pidMark += 1
          pidMark
        } else {
          tokens.head.toLong
        }
        val edger = toEdge(gen, docId, numTopics, reverse)_
        tokens.tail.map(field => {
          val pairs = field.split(":")
          val termId = pairs(0).toInt
          val termCnt = if (pairs.length > 1) pairs(1).toInt else 1
          (termId, termCnt)
        }).filter(_._2 > 0).map(edger)
      })
    })
  }

  def convertBowDocs(bowDocs: RDD[BOW],
    numTopics: Int,
    ignDid: Boolean,
    reverse: Boolean): RDD[Edge[TA]] = {
    bowDocs.mapPartitionsWithIndex((pid, iter) => {
      val gen = new XORShiftRandom(pid + 117)
      var pidMark = pid.toLong << 48
      iter.flatMap(Function.tupled((oDocId, tokens) => {
        val docId = if (ignDid) {
          pidMark += 1
          pidMark
        } else {
          oDocId
        }
        val edger = toEdge(gen, docId, numTopics, reverse)_
        tokens.activeIterator.filter(_._2 > 0).map(edger)
      }))
    })
  }

  private def toEdge(gen: Random,
    docId: Long,
    numTopics: Int,
    reverse: Boolean)
    (termPair: (Int, Count)): Edge[TA] = {
    val (termId, termCnt) = termPair
    val topics = Array.fill(termCnt)(gen.nextInt(numTopics))
    if (!reverse) {
      Edge(termId, genNewDocId(docId), topics)
    } else {
      Edge(genNewDocId(docId), termId, topics)
    }
  }

  def resampleCorpus(corpus: Graph[TC, TA],
    numTopics: Int,
    storageLevel: StorageLevel): Graph[TC, TA] = {
    val conf = corpus.edges.context.getConf
    conf.get(cs_initStrategy, "random") match {
      case "sparse" =>
        corpus.persist(storageLevel)
        val gen = new XORShiftRandom()
        val tMin = math.min(1000, numTopics / 100)
        val degGraph = GraphImpl(corpus.degrees, corpus.edges)
        val reSampledGraph = degGraph.mapVertices((vid, deg) => {
          if (isTermId(vid) && deg > tMin) {
            Array.fill(tMin)(gen.nextInt(numTopics))
          } else {
            null
          }
        }).mapTriplets((pid, iter) => {
          val gen = new XORShiftRandom(pid + 223)
          iter.map(triplet => {
            val wc = triplet.srcAttr
            val topics = triplet.attr
            if (wc == null) {
              topics
            } else {
              val tSize = wc.length
              topics.map(_ => wc(gen.nextInt(tSize)))
            }
          })
        }, TripletFields.Src)
        GraphImpl(corpus.vertices, reSampledGraph.edges)
      case _ =>
        corpus
    }
  }
}
