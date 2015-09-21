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

  def updateVertexCounters(newCorpus: Graph[TC, TA],
    inferenceOnly: Boolean = false): Unit = {
    val numThreads = scConf.getInt(cs_numThreads, 1)
    val graph = newCorpus.asInstanceOf[GraphImpl[TC, TA]]
    val vertices = graph.vertices
    val edges = graph.edges
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
          for (topic <- topics) {
            if (!inferenceOnly) termTopics match {
              case v: BDV[Count] => v(topic) += 1
              case v: BSV[Count] =>
                v(topic) += 1
                if (v.activeSize > (numTopics >> 2)) {
                  termTuple = (l2g(si), toBDV(v))
                  results(si) = termTuple
                  termTopics = termTuple._2
                }
            }
            docTopics.synchronized {
              docTopics(topic) += 1
            }
          }
          pos += 1
        }
      }))
      Await.ready(all, 1.hour)
      es.shutdown()

      results.par.filter(_ != null)
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
                    if (u.activeSize > (numTopics >> 2)) toBDV(u) else u
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
    corpus = GraphImpl.fromExistingRDDs(vertices.withPartitionsRDD(partRDD), edges)
  }

  def collectTopicCounters(): Unit = {
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
    val count = gtc.activeValuesIterator.map(_.toLong).sum
    assert(count == numTokens, s"numTokens=$numTokens, count=$count")
    topicCounters = gtc
  }

  def runGibbsSampling(totalIter: Int): Unit = {
    val pplx = scConf.getBoolean(cs_calcPerplexity, false)
    val saveIntv = scConf.getInt(cs_saveInterval, 0)
    if (pplx) {
      println("Before Gibbs sampling:")
      LDAPerplexity.output(this, println)
    }
    for (iter <- 1 to totalIter) {
      println(s"Start Gibbs sampling (Iteration $iter/$totalIter)")
      val startedAt = System.nanoTime()
      gibbsSampling(iter)
      if (pplx) {
        LDAPerplexity.output(this, println)
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
      println(s"End Gibbs sampling (Iteration $iter/$totalIter) takes: $elapsedSeconds secs")
    }
  }

  def gibbsSampling(sampIter: Int, inferenceOnly: Boolean = false): Unit = {
    val chkptIntv = scConf.getInt(cs_chkptInterval, 0)
    val prevCorpus = corpus
    val sampledCorpus = algo.sampleGraph(corpus, topicCounters, sampIter + seed,
      numTokens, numTopics, numTerms, alpha, alphaAS, beta)
    updateVertexCounters(sampledCorpus, inferenceOnly)
    if (chkptIntv > 0 && sampIter % chkptIntv == 1) {
      if (corpus.edges.context.getCheckpointDir.isDefined) {
        corpus.checkpoint()
      }
    }
    corpus.persist(storageLevel)
    corpus.edges.setName(s"edges-$sampIter").count()
    corpus.vertices.setName(s"vertices-$sampIter")
    collectTopicCounters()
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
    for (iter <- 1 to runIter) {
      println(s"Save TopicModel (Iteration $iter/$runIter)")
      gibbsSampling(iter, inferenceOnly)
      countersSum = countersSum.innerZipJoin(vertices)((_, a, b) => a :+= b)
      countersSum.persist(storageLevel)
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
      updateVertexCounters(mergingCorpus)
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
    val edges = initCorpus.edges.setName("edges-0")
    val vertices = initCorpus.vertices
    val numTokens = edges.map(_.attr.length.toLong).reduce(_ + _)
    println(s"tokens in the corpus: $numTokens")
    val numTerms = vertices.map(_._1).filter(isTermId).count().toInt
    println(s"terms in the corpus: $numTerms")
    val numDocs = vertices.map(_._1).filter(isDocId).count()
    println(s"docs in the corpus: $numDocs")
    docs.unpersist(blocking=false)
    val lda = new LDA(initCorpus, numTopics, numTerms, numDocs, numTokens, alpha, beta, alphaAS,
      algo, storageLevel)
    lda.updateVertexCounters(initCorpus)
    lda.getCorpus.vertices.persist(storageLevel).setName("vertices-0")
    lda.collectTopicCounters()
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
    initCorpus.edges.setName("edges-0")
    val vertices = initCorpus.vertices
    val numDocs = vertices.map(_._1).filter(isDocId).count()
    println(s"docs in the corpus: $numDocs")
    docs.unpersist(blocking=false)
    val lda = new LDA(initCorpus, numTopics, numTerms, numDocs, numTokens, alpha, beta, alphaAS,
      algo, storageLevel)
    lda.updateVertexCounters(initCorpus)
    lda.getCorpus.joinVertices(computedModel.termTopicsRDD)((_, _, computedCounter) => computedCounter)
    lda.getCorpus.vertices.persist(storageLevel).setName("vertices-0")
    lda.collectTopicCounters()
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
    partCorpus.persist(storageLevel)
    val reCorpus = resampleCorpus(partCorpus, numTopics)
    val edges = reCorpus.edges
    val numEdges = edges.persist(storageLevel).count()
    println(s"edges in the corpus: $numEdges")
    initCorpus.unpersist(blocking=false)
    edges
  }

  def resampleCorpus(corpus: Graph[TC, TA], numTopics: Int): Graph[TC, TA] = {
    val conf = corpus.edges.context.getConf
    conf.get(cs_initStrategy, "random") match {
      case "sparse" =>
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

  def convertRawDocs(rawDocs: RDD[String], numTopics: Int): RDD[Edge[TA]] = {
    rawDocs.mapPartitionsWithIndex((pid, iter) => {
      val gen = new XORShiftRandom(pid + 117)
      iter.flatMap(line => {
        val tokens = line.split(raw"\t|\s+")
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
}
