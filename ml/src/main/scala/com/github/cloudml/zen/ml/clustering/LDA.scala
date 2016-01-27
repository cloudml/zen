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

import breeze.linalg.{DenseVector => BDV}
import com.github.cloudml.zen.ml.clustering.LDADefines._
import com.github.cloudml.zen.ml.clustering.algorithm.LDATrainer
import com.github.cloudml.zen.ml.partitioner._
import com.github.cloudml.zen.ml.util._
import org.apache.hadoop.fs.Path
import org.apache.spark.graphx2._
import org.apache.spark.graphx2.impl._
import org.apache.spark.mllib.linalg.distributed.{MatrixEntry, RowMatrix}
import org.apache.spark.mllib.linalg.{SparseVector => SSV, Vector => SV}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel


class LDA(@transient var edges: EdgeRDDImpl[TA, _],
  @transient var verts: VertexRDDImpl[TC],
  val numTopics: Int,
  val numTerms: Int,
  val numDocs: Long,
  val numTokens: Long,
  var alpha: Double,
  var beta: Double,
  var alphaAS: Double,
  val algo: LDATrainer,
  var storageLevel: StorageLevel) extends Serializable {

  @transient var topicCounters: BDV[Count] = _
  @transient lazy val seed = (new XORShiftRandom).nextInt()
  @transient var edgeCpFile: String = _
  @transient var vertCpFile: String = _

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

  def termVertices: VertexRDD[TC] = verts.filter(t => isTermId(t._1))

  def docVertices: VertexRDD[TC] = verts.filter(t => isDocId(t._1))

  @inline private def scContext = edges.context

  @inline private def scConf = scContext.getConf

  def init(computedModel: Option[RDD[NwkPair]] = None): Unit = {
    val initPartRDD = edges.partitionsRDD.mapPartitions(_.map(Function.tupled((pid, ep) => {
      (pid, algo.initEdgePartition(ep))
    })), preservesPartitioning=true)
    val newEdges = edges.withPartitionsRDD(initPartRDD)
    newEdges.persist(storageLevel).setName("edges-0")
    edges = newEdges

    verts = algo.updateVertexCounters(newEdges, verts)
    verts = computedModel match {
      case Some(cm) =>
        val ccm = compressCounterRDD(cm, numTopics)
        verts.leftJoin(ccm)((_, uc, cc) => cc.getOrElse(uc)).asInstanceOf[VertexRDDImpl[TC]]
      case None => verts
    }
    verts.persist(storageLevel).setName("vertices-0")
    topicCounters = algo.collectTopicCounters(verts)
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
      val startedAt = System.nanoTime
      gibbsSampling(iter)
      if (pplx) {
        LDAPerplexity(this).output(println)
      }
      if (saveIntv > 0 && iter % saveIntv == 0 && iter < totalIter) {
        val model = toLDAModel
        val savPath = new Path(scConf.get(cs_outputpath) + s"-iter$iter")
        val fs = SparkUtils.getFileSystem(scConf, savPath)
        fs.delete(savPath, true)
        model.save(scContext, savPath.toString)
        println(s"Model saved after Iteration $iter")
      }
      val elapsedSeconds = (System.nanoTime - startedAt) / 1e9
      println(s"End Gibbs sampling (Iteration $iter/$totalIter) takes total: $elapsedSeconds secs")
      iter += 1
    }
  }

  def gibbsSampling(sampIter: Int): Unit = {
    val chkptIntv = scConf.getInt(cs_chkptInterval, 0)
    val needChkpt = chkptIntv > 0 && sampIter % chkptIntv == 1 && scContext.getCheckpointDir.isDefined
    val startedAt = System.nanoTime

    val newEdges = algo.sampleGraph(edges, verts, topicCounters, seed, sampIter,
      numTokens, numTerms, alpha, alphaAS, beta)
    newEdges.persist(storageLevel).setName(s"edges-$sampIter")
    if (needChkpt) {
      newEdges.checkpoint()
      newEdges.partitionsRDD.count()
    }

    val newVerts = algo.updateVertexCounters(newEdges, verts)
    newVerts.persist(storageLevel).setName(s"vertices-$sampIter")
    if (needChkpt) {
      newVerts.checkpoint()
    }
    topicCounters = algo.collectTopicCounters(newVerts)
    val count = topicCounters.data.par.map(_.toLong).sum
    assert(count == numTokens, s"numTokens=$numTokens, count=$count")
    edges.unpersist(blocking=false)
    verts.unpersist(blocking=false)
    edges = newEdges
    verts = newVerts

    if (needChkpt) {
      if (edgeCpFile != null && vertCpFile != null) {
        SparkUtils.deleteChkptDirs(scConf, Array(edgeCpFile, vertCpFile))
      }
      edgeCpFile = newEdges.getCheckpointFile.get
      vertCpFile = newVerts.getCheckpointFile.get
    }
    val elapsedSeconds = (System.nanoTime - startedAt) / 1e9
    println(s"Sampling & update paras $sampIter takes: $elapsedSeconds secs")
  }

  def toLDAModel: DistributedLDAModel = {
    val termTopicsRDD = decompressVertexRDD(termVertices, numTopics)
    termTopicsRDD.persist(storageLevel)
    new DistributedLDAModel(termTopicsRDD, numTopics, numTerms, numTokens, alpha, beta, alphaAS, storageLevel)
  }

//  /**
//   * run more iters, return averaged counters
//   * @param filter return which vertices
//   * @param runIter saved more these iters' averaged model
//   */
//  def runSum(filter: VertexId => Boolean,
//    runIter: Int = 0): RDD[(VertexId, TC)] = {
//    def vertices = verts.filter(t => filter(t._1))
//    var countersSum = vertices
//    countersSum.persist(storageLevel)
//    var iter = 1
//    while (iter <= runIter) {
//      println(s"Save TopicModel (Iteration $iter/$runIter)")
//      gibbsSampling(iter)
//      countersSum = countersSum.innerZipJoin(vertices)((_, a, b) => a :+= b)
//      countersSum.persist(storageLevel)
//      iter += 1
//    }
//    countersSum
//  }

//  def mergeDuplicateTopic(threshold: Double = 0.95D): Map[Int, Int] = {
//    val rows = termVertices.map(t => t._2).map(v => {
//      val length = v.length
//      val index = v.activeKeysIterator.toArray
//      val data = v.activeValuesIterator.toArray.map(_.toDouble)
//      new SSV(length, index, data).asInstanceOf[SV]
//    })
//    val simMatrix = new RowMatrix(rows).columnSimilarities()
//    val minMap = simMatrix.entries.filter {
//      case MatrixEntry(row, column, sim) => sim > threshold && row != column
//    }.map {
//      case MatrixEntry(row, column, sim) => (column.toInt, row.toInt)
//    }.groupByKey().map {
//      case (topic, simTopics) => (topic, simTopics.min)
//    }.collect().toMap
//    if (minMap.nonEmpty) {
//      val mergingCorpus = corpus.mapEdges(_.attr.map(topic =>
//        minMap.getOrElse(topic, topic))
//      )
//      corpus = algo.updateVertexCounters(mergingCorpus, numTopics)
//    }
//    minMap
//  }
}

object LDA {
  def apply(docs: EdgeRDD[TA],
    numTopics: Int,
    alpha: Double,
    beta: Double,
    alphaAS: Double,
    algo: LDATrainer,
    storageLevel: StorageLevel): LDA = {
    val initCorpus = LBVertexRDDBuilder.fromEdgeRDD[TC, TA](docs, storageLevel)
    initCorpus.persist(storageLevel)
    val edges = initCorpus.edges
    val numTokens = edges.count()
    println(s"tokens in the corpus: $numTokens")
    val verts = initCorpus.vertices.asInstanceOf[VertexRDDImpl[TC]]
    val numTerms = verts.map(_._1).filter(isTermId).count().toInt
    println(s"terms in the corpus: $numTerms")
    val numDocs = verts.map(_._1).filter(isDocId).count()
    println(s"docs in the corpus: $numDocs")
    val lda = new LDA(edges, verts, numTopics, numTerms, numDocs, numTokens, alpha, beta, alphaAS,
      algo, storageLevel) { init() }
    initCorpus.unpersist(blocking=false)
    lda
  }

  // initialize LDA for inference or incremental training
  def apply(computedModel: DistributedLDAModel,
    docs: EdgeRDD[TA],
    algo: LDATrainer): LDA = {
    val numTopics = computedModel.numTopics
    val numTerms = computedModel.numTerms
    val numTokens = computedModel.numTokens
    val alpha = computedModel.alpha
    val beta = computedModel.beta
    val alphaAS = computedModel.alphaAS
    val storageLevel = computedModel.storageLevel
    println(s"tokens in the corpus: $numTokens")
    println(s"terms in the corpus: $numTerms")
    val initCorpus = LBVertexRDDBuilder.fromEdgeRDD[TC, TA](docs, storageLevel)
    initCorpus.persist(storageLevel)
    val edges = initCorpus.edges
    val verts = initCorpus.vertices.asInstanceOf[VertexRDDImpl[TC]]
    val numDocs = verts.map(_._1).filter(isDocId).count()
    println(s"docs in the corpus: $numDocs")
    val lda = new LDA(edges, verts, numTopics, numTerms, numDocs, numTokens, alpha, beta, alphaAS,
      algo, storageLevel) { init(Some(computedModel.termTopicsRDD)) }
    verts.unpersist(blocking=false)
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
   * @param algo       LDA training algorithm used
   * @param storageLevel StorageLevel that the LDA Model RDD uses
   * @return DistributedLDAModel
   */
  def train(docs: EdgeRDD[TA],
    totalIter: Int,
    numTopics: Int,
    alpha: Double,
    beta: Double,
    alphaAS: Double,
    algo: LDATrainer,
    storageLevel: StorageLevel): DistributedLDAModel = {
    val lda = LDA(docs, numTopics, alpha, beta, alphaAS, algo, storageLevel)
    lda.runGibbsSampling(totalIter)
    lda.toLDAModel
  }

  def incrementalTrain(docs: EdgeRDD[TA],
    computedModel: DistributedLDAModel,
    totalIter: Int,
    algo: LDATrainer,
    storageLevel: StorageLevel): DistributedLDAModel = {
    val lda = LDA(computedModel, docs, algo)
    var iter = 1
    while (iter <= 15) {
      lda.gibbsSampling(iter)
      iter += 1
    }
    lda.runGibbsSampling(totalIter)
    lda.toLDAModel
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
    algo: LDATrainer,
    storageLevel: StorageLevel): EdgeRDD[TA] = {
    val conf = orgDocs.context.getConf
    val ignDid = conf.getBoolean(cs_ignoreDocId, false)
    val partStrategy = conf.get(cs_partStrategy, "dbh")
    val initStrategy = conf.get(cs_initStrategy, "random")
    val byDoc = algo.isByDoc
    val docs = docType match {
      case "raw" => convertRawDocs(orgDocs.asInstanceOf[RDD[String]], numTopics, ignDid, byDoc)
      case "bow" => convertBowDocs(orgDocs.asInstanceOf[RDD[BOW]], numTopics, ignDid, byDoc)
    }
    val graph: Graph[TC, TA] = LBVertexRDDBuilder.fromEdges(docs, storageLevel)
    graph.persist(storageLevel)
    graph.edges.setName("rawEdges").count()

    val partCorpus = partitionCorpus(graph, partStrategy, byDoc, storageLevel)
    val initCorpus = reinitCorpus(partCorpus, initStrategy, numTopics, storageLevel)
    val edges = initCorpus.edges
    edges.persist(storageLevel).count()
    graph.unpersist(blocking=false)
    edges
  }

  def convertRawDocs(rawDocs: RDD[String],
    numTopics: Int,
    ignDid: Boolean,
    byDoc: Boolean): RDD[Edge[TA]] = {
    rawDocs.mapPartitionsWithIndex((pid, iter) => {
      val gen = new XORShiftRandom(pid + 117)
      var pidMark = pid.toLong << 48
      iter.flatMap(line => {
        val tokens = line.split(raw"\t|\s+").view
        val docId = if (ignDid) {
          pidMark += 1
          pidMark
        } else {
          tokens.head.toLong
        }
        tokens.tail.flatMap(field => {
          val pairs = field.split(":")
          val termId = pairs(0).toInt
          val termCnt = if (pairs.length > 1) pairs(1).toInt else 1
          if (termCnt > 0) {
            Range(0, termCnt).map(_ => if (byDoc) {
              Edge(genNewDocId(docId), termId, gen.nextInt(numTopics))
            } else {
              Edge(termId, genNewDocId(docId), gen.nextInt(numTopics))
            })
          } else {
            Iterator.empty
          }
        })
      })
    })
  }

  def convertBowDocs(bowDocs: RDD[BOW],
    numTopics: Int,
    ignDid: Boolean,
    byDoc: Boolean): RDD[Edge[TA]] = {
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
        tokens.activeIterator.filter(_._2 > 0).flatMap(Function.tupled((termId, termCnt) =>
          Range(0, termCnt).map(_ => if (byDoc) {
            Edge(genNewDocId(docId), termId, gen.nextInt(numTopics))
          } else {
            Edge(termId, genNewDocId(docId), gen.nextInt(numTopics))
          })
        ))
      }))
    })
  }

  def partitionCorpus(corpus: Graph[TC, TA],
    partStrategy: String,
    byDoc: Boolean,
    storageLevel: StorageLevel): Graph[TC, TA] = partStrategy match {
    case "direct" =>
      println("don't repartition, directly build graph.")
      corpus
    case "byterm" =>
      println("partition corpus by terms.")
      if (byDoc) {
        EdgeDstPartitioner.partitionByEDP[TC, TA](corpus, storageLevel)
      } else {
        corpus.partitionBy(PartitionStrategy.EdgePartition1D)
      }
    case "bydoc" =>
      println("partition corpus by docs.")
      if (byDoc) {
        corpus.partitionBy(PartitionStrategy.EdgePartition1D)
      } else {
        EdgeDstPartitioner.partitionByEDP[TC, TA](corpus, storageLevel)
      }
    case "edge2d" =>
      println("using Edge2D partition strategy.")
      corpus.partitionBy(PartitionStrategy.EdgePartition2D)
    case "dbh" =>
      println("using Degree-based Hashing partition strategy.")
      DBHPartitioner.partitionByDBH[TC, TA](corpus, storageLevel)
    case "vsdlp" =>
      println("using Vertex-cut Stochastic Dynamic Label Propagation partition strategy.")
      VSDLPPartitioner.partitionByVSDLP[TC, TA](corpus, 4, storageLevel)
    case "bbr" =>
      println("using Bounded & Balanced Rearranger partition strategy.")
      BBRPartitioner.partitionByBBR[TC, TA](corpus, storageLevel)
    case _ =>
      throw new NoSuchMethodException("No this algorithm or not implemented.")
  }

  def reinitCorpus(corpus: Graph[TC, TA],
    initStrategy: String,
    numTopics: Int,
    storageLevel: StorageLevel): Graph[TC, TA] = initStrategy match {
    case "random" =>
      println("fully randomized initialization.")
      corpus
    case "sparse" =>
      println("sparsely init on terms.")
      corpus.persist(storageLevel)
      val gen = new XORShiftRandom
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
          if (wc == null) {
            triplet.attr
          } else {
            wc(gen.nextInt(wc.length))
          }
        })
      }, TripletFields.Src)
      GraphImpl(corpus.vertices, reSampledGraph.edges)
    case _ =>
      throw new NoSuchMethodException("No this algorithm or not implemented.")
  }
}
