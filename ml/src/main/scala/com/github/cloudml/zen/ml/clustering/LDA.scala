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
import java.util.concurrent.atomic.AtomicIntegerArray
import scala.concurrent._
import scala.concurrent.duration.Duration

import LDA._
import LDADefines._
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV}
import com.github.cloudml.zen.ml.partitioner._
import com.github.cloudml.zen.ml.util.XORShiftRandom
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

  @transient var seed = new XORShiftRandom().nextInt()
  @transient var totalTopicCounter = collectTopicCounter()

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

  def getCorpus: Graph[TC, TA] = corpus

  def termVertices: VertexRDD[TC] = corpus.vertices.filter(t => !isDocId(t._1))

  def docVertices: VertexRDD[TC] = corpus.vertices.filter(t => isDocId(t._1))

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
      val perplexity = LDAMetrics.perplexity(this)
      println(s"Before Gibbs sampling: perplexity=$perplexity")
    }
    for (iter <- 1 to totalIter) {
      println(s"Start Gibbs sampling (Iteration $iter/$totalIter)")
      val startedAt = System.nanoTime()
      gibbsSampling(iter)
      if (pplx) {
        val perplexity = LDAMetrics.perplexity(this)
        println(s"Gibbs sampling (Iteration $iter/$totalIter): perplexity=$perplexity")
      }
      if (saveIntv > 0 && iter % saveIntv == 0) {
        val model = toLDAModel()
        val outputPath = scConf.get(cs_outputpath)
        model.save(sc, s"$outputPath-iter$iter", isTransposed=true)
        println(s"Model saved after Iteration $iter")
      }
      val elapsedSeconds = (System.nanoTime() - startedAt) / 1e9
      println(s"End Gibbs sampling (Iteration $iter/$totalIter) takes: $elapsedSeconds secs")
    }
  }

  def gibbsSampling(sampIter: Int, inferenceOnly: Boolean = false): Unit = {
    val sc = corpus.edges.context
    val chkptIntv = scConf.getInt(cs_chkptInterval, 0)
    val prevCorpus = corpus
    val sampledCorpus = algo.sampleGraph(corpus, totalTopicCounter, sampIter + seed,
      numTokens, numTopics, numTerms, alpha, alphaAS, beta)
    corpus = updateVertexCounters(sampledCorpus, numTopics, inferenceOnly)
    if (chkptIntv > 0 && sampIter % chkptIntv == 1 && sc.getCheckpointDir.isDefined) {
      corpus.checkpoint()
    }
    corpus.persist(storageLevel)
    corpus.edges.setName(s"edges-$sampIter").count()
    corpus.vertices.setName(s"vertices-$sampIter")
    totalTopicCounter = collectTopicCounter()
    prevCorpus.unpersist(blocking=false)
  }

  /**
   * run more iters, return averaged counters
   * @param filter return which vertices
   * @param runIter saved more these iters' averaged model
   */
  def runSum(filter: VertexId => Boolean,
    runIter: Int = 0,
    inferenceOnly: Boolean = false): RDD[(VertexId, BSV[Double])] = {
    @inline def vertices = corpus.vertices.filter(t => filter(t._1))
    var countersSum: RDD[(VertexId, BSV[Double])] = vertices.map(t => (t._1, t._2.mapValues(_.toDouble)))
    countersSum.persist(storageLevel).count()
    for (iter <- 1 to runIter) {
      println(s"Save TopicModel (Iteration $iter/$runIter)")
      gibbsSampling(iter, inferenceOnly)
      val newCounterSum = countersSum.join(vertices).map {
        case (term, (a, b)) => (term, a :+= b.mapValues(_.toDouble))
      }
      newCounterSum.persist(storageLevel).count()
      countersSum.unpersist(blocking=false)
      countersSum = newCounterSum
    }
    val counters = if (runIter == 0) {
      countersSum
    } else {
      val aver = countersSum.mapValues(_ /= (runIter + 1).toDouble)
      aver.persist(storageLevel).count()
      countersSum.unpersist(blocking=false)
      aver
    }
    counters
  }

  def toLDAModel(runIter: Int = 0): DistributedLDAModel = {
    val gen = new XORShiftRandom()
    val ttcs = runSum(isTermId, runIter).mapValues(_.mapValues(v => {
      val l = math.floor(v)
      if (gen.nextDouble() > v - l) l else l + 1
    }.toInt))
    new DistributedLDAModel(ttcs, numTopics, numTerms, numTokens, alpha, beta, alphaAS, storageLevel)
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
    val initCorpus = LBVertexRDDBuilder.fromEdgeRDD[TC, TA](docs, storageLevel)
    val edges = initCorpus.edges
    edges.setName("edges-0").persist(storageLevel)
    val numTokens = edges.map(_.attr.length.toLong).reduce(_ + _)
    println(s"tokens in the corpus: $numTokens")
    val corpus = updateVertexCounters(initCorpus, numTopics)
    val vertices = corpus.vertices
    vertices.setName("vertices-0").persist(storageLevel)
    val numTerms = vertices.filter(t => isTermId(t._1)).count().toInt
    println(s"terms in the corpus: $numTerms")
    val numDocs = vertices.filter(t => isDocId(t._1)).count()
    println(s"docs in the corpus: $numDocs")
    new LDA(corpus, numTopics, numTerms, numDocs, numTokens, alpha, beta, alphaAS, algo, storageLevel)
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
    val initCorpus = LBVertexRDDBuilder.fromEdgeRDD[TC, TA](docs, storageLevel)
    initCorpus.edges.setName("edges-0").persist(storageLevel)
    val corpus = updateVertexCounters(initCorpus, numTopics)
      .joinVertices(computedModel.termTopicCounters)((_, _, computedCounter) => computedCounter)
    val vertices = corpus.vertices
    vertices.setName("vertices-0").persist(storageLevel)
    val numDocs = vertices.filter(t => isDocId(t._1)).count()
    println(s"docs in the corpus: $numDocs")
    new LDA(corpus, numTopics, numTerms, numDocs, numTokens, alpha, beta, alphaAS, algo, storageLevel)
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
      case _ =>
        throw new NoSuchMethodException("No this algorithm or not implemented.")
    }
    val lda = LDA(computedModel, docs, algo)
    for (iter <- 1 to 15) {
      lda.gibbsSampling(iter, inferenceOnly=true)
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
    storageLevel: StorageLevel): EdgeRDD[TA] = {
    val conf = orgDocs.context.getConf
    val docs = docType match {
      case "raw" => convertRawDocs(orgDocs.asInstanceOf[RDD[String]], numTopics)
      case "bow" => convertBowDocs(orgDocs.asInstanceOf[RDD[BOW]], numTopics)
    }
    val initCorpus: Graph[TC, TA] = LBVertexRDDBuilder.fromEdges(docs, storageLevel)
    initCorpus.persist(storageLevel)
    val partCorpus = conf.get(cs_partStrategy, "dbh") match {
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
    val edges = partCorpus.edges
    val numEdges = edges.persist(storageLevel).count()
    println(s"edges in the corpus: $numEdges")
    initCorpus.unpersist(blocking=false)
    edges
  }

  def convertRawDocs(rawDocs: RDD[String], numTopics: Int): RDD[Edge[TA]] = {
    rawDocs.mapPartitionsWithIndex((pid, iter) => {
      val gen = new XORShiftRandom(pid + 117)
      iter.flatMap(line => {
        val tokens = line.split("\\t|\\s+")
        val docId = tokens.head.toLong
        val edger = toEdge(gen, docId, numTopics) _
        tokens.tail.map(field => {
          val pairs = field.split(":")
          val termId = pairs(0).toInt
          val termCnt = if (pairs.length > 1) pairs(1).toInt else 1
          (termId, termCnt)
        }).filter(_._2 > 0).map(edger)
      })
    })
  }

  def convertBowDocs(bowDocs: RDD[BOW], numTopics: Int): RDD[Edge[TA]] = {
    bowDocs.mapPartitionsWithIndex((pid, iter) => {
      val gen = new XORShiftRandom(pid + 117)
      iter.flatMap{
        case (docId, tokens) =>
          val edger = toEdge(gen, docId, numTopics) _
          tokens.activeIterator.filter(_._2 > 0).map(edger)
      }
    })
  }

  private def toEdge(gen: Random, docId: Long, numTopics: Int)
    (termPair: (Int, Count)): Edge[TA] = {
    val (termId, termCnt) = termPair
    val topics = Array.fill(termCnt)(gen.nextInt(numTopics))
    Edge(termId, genNewDocId(docId), topics)
  }

  private def updateVertexCounters(corpus: Graph[TC, TA],
    numTopics: Int,
    inferenceOnly: Boolean = false): GraphImpl[TC, TA] = {
    val graph = corpus.asInstanceOf[GraphImpl[TC, TA]]
    val vertices = graph.vertices
    val edges = graph.edges
    val numThreads = edges.context.getConf.getInt(cs_numThreads, 1)
    val shippedCounters = edges.partitionsRDD.mapPartitions(_.flatMap(t => {
      val ep = t._2
      val totalSize = ep.size
      val lcSrcIds = ep.localSrcIds
      val lcDstIds = ep.localDstIds
      val l2g = ep.local2global
      val vattrs = ep.vertexAttrs
      val data = ep.data
      val vertSize = vattrs.length
      val results = new Array[(VertexId, TC)](vertSize)
      val marks = new AtomicIntegerArray(vertSize)

      implicit val ec = ExecutionContext.fromExecutorService(Executors.newFixedThreadPool(numThreads))
      val all = Future.traverse(ep.index.iterator)(t => Future {
        var pos = t._2
        val lcVid = lcSrcIds(pos)
        var termTuple = results(lcVid)
        if (termTuple == null && !inferenceOnly) {
          termTuple = (l2g(lcVid), BSV.zeros[Count](numTopics))
          results(lcVid) = termTuple
        }
        val termTopicCounter = termTuple._2
        while (pos < totalSize && lcSrcIds(pos) == lcVid) {
          val di = lcDstIds(pos)
          var docTuple = results(di)
          if (docTuple == null) {
            if (marks.getAndDecrement(di) == 0) {
              docTuple = (l2g(di), BSV.zeros[Count](numTopics))
              results(di) = docTuple
              marks.set(di, Int.MaxValue)
            } else {
              while (marks.get(di) < 0) {}
              docTuple = results(di)
            }
          }
          val docTopicCounter = docTuple._2
          val topics = data(pos)
          for (t <- topics) {
            if (!inferenceOnly) {
              termTopicCounter(t) += 1
            }
            docTopicCounter.synchronized {
              docTopicCounter(t) += 1
            }
          }
          pos += 1
        }
      })
      Await.ready(all, Duration.Inf)
      ec.shutdown()

      results.filter(_ != null)
    })).partitionBy(vertices.partitioner.get)

    val partRDD = vertices.partitionsRDD.zipPartitions(shippedCounters, preservesPartitioning=true)(
      (svpIter, cntsIter) => svpIter.map(svp => {
        val results = svp.values
        val index = svp.index
        val marks = new AtomicIntegerArray(results.length)
        implicit val ec = ExecutionContext.fromExecutorService(Executors.newFixedThreadPool(numThreads))
        val all = Future.traverse(cntsIter)(cnts => Future {
          val (vid, counter) = cnts
          val i = index.getPos(vid)
          if (marks.getAndDecrement(i) == 0) {
            results(i) = counter
          } else {
            while (marks.get(i) < 0) {}
            val agg = results(i)
            agg.synchronized {
              agg :+= counter
            }
          }
          marks.set(i, Int.MaxValue)
        })
        Await.ready(all, Duration.Inf)
        ec.shutdown()
        svp.withValues(results)
      })
    )
    GraphImpl.fromExistingRDDs(vertices.withPartitionsRDD(partRDD), edges)
  }
}
