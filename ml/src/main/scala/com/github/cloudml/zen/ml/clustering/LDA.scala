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
import java.util.concurrent.{ConcurrentLinkedQueue, CountDownLatch}
import java.util.concurrent.atomic.AtomicIntegerArray

import LDA._
import LDADefines._
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, sum}
import com.github.cloudml.zen.ml.partitioner._
import com.github.cloudml.zen.ml.util.XORShiftRandom
import org.apache.log4j.Logger
import org.apache.spark.{SparkContext, SparkConf}
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
  def apply(docs: RDD[Edge[TA]],
    numTopics: Int,
    alpha: Double,
    beta: Double,
    alphaAS: Double,
    algo: LDAAlgorithm,
    storageLevel: StorageLevel): LDA = {
    val initCorpus = LBVertexRDDBuilder.fromEdges[TC, TA](docs, storageLevel)
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
    docs: RDD[Edge[TA]],
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
    val initCorpus = LBVertexRDDBuilder.fromEdges[TC, TA](docs, storageLevel)
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
   * @param docs       RDD of corpus edges
   * @param totalIter  the number of iterations
   * @param numTopics  the number of topics (5000+ for large data)
   * @param alpha      recommend to be (5.0 /numTopics)
   * @param beta       recommend to be in range 0.001 - 0.1
   * @param alphaAS    recommend to be in range 0.01 - 1.0
   * @param storageLevel StorageLevel that the LDA Model RDD uses
   * @return DistributedLDAModel
   */
  def train(docs: RDD[Edge[TA]],
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

  def incrementalTrain(docs: RDD[Edge[TA]],
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
   * @param bowDocs  RDD of documents, which are term (word) count vectors paired with IDs.
   *                 The term count vectors are "bags of words" with a fixed-size vocabulary
   *                 (where the vocabulary size is the length of the vector).
   *                 Document IDs must be unique and >= 0.
   */
  def initializeCorpusEdges(bowDocs: RDD[BOW],
    numTopics: Int,
    storageLevel: StorageLevel): EdgeRDD[TA] = {
    val conf = bowDocs.context.getConf
    val edges = bowDocs.mapPartitionsWithIndex((pid, iter) => {
      val gen = new XORShiftRandom(pid + 117)
      iter.flatMap {
        case (docId, doc) => initializeEdges(gen, doc, docId, numTopics)
      }
    })
    val initCorpus: Graph[TC, TA] = LBVertexRDDBuilder.fromEdges(edges, storageLevel)
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
    val corpusEdges = partCorpus.edges
    val numEdges = corpusEdges.persist(storageLevel).count()
    println(s"edges in the corpus: $numEdges")
    bowDocs.unpersist(blocking=false)
    initCorpus.unpersist(blocking=false)
    corpusEdges
  }

  private def initializeEdges(
    gen: Random,
    doc: BSV[Int],
    docId: DocId,
    numTopics: Int): Iterator[Edge[TA]] = {
    val newDocId: DocId = genNewDocId(docId)
    doc.activeIterator.filter(_._2 > 0).map {
      case (termId, counter) =>
        val topics = new Array[Int](counter)
        for (i <- 0 until counter) {
          topics(i) = gen.nextInt(numTopics)
        }
        Edge(termId, newDocId, topics)
    }
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
      val sizePerThrd = {
        val npt = totalSize / numThreads
        if (npt * numThreads == totalSize) npt else npt + 1
      }
      val vertSize = ep.vertexAttrs.length
      val results = new Array[(VertexId, TC)](vertSize)
      val marks = new AtomicIntegerArray(vertSize)
      val doneSignal = new CountDownLatch(numThreads)
      val threads = new Array[Thread](numThreads)
      for (threadId <- threads.indices) {
        threads(threadId) = new Thread(new Runnable {
          val startPos = sizePerThrd * threadId
          val endPos = math.min(sizePerThrd * (threadId + 1), totalSize)

          override def run(): Unit = {
            val logger = Logger.getLogger(this.getClass.getName)
            val lcSrcIds = ep.localSrcIds
            val lcDstIds = ep.localDstIds
            val l2g = ep.local2global
            val data = ep.data
            try {
              if (inferenceOnly) {
                for (i <- startPos until endPos) {
                  val di = lcDstIds(i)
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
                  val topics = data(i)
                  for (t <- topics) {
                    docTopicCounter.synchronized {
                      docTopicCounter(t) += 1
                    }
                  }
                }
              } else {
                for (i <- startPos until endPos) {
                  val si = lcSrcIds(i)
                  val di = lcDstIds(i)
                  var termTuple = results(si)
                  if (termTuple == null) {
                    if (marks.getAndDecrement(si) == 0) {
                      termTuple = (l2g(si), BSV.zeros[Count](numTopics))
                      results(si) = termTuple
                      marks.set(si, Int.MaxValue)
                    } else {
                      while (marks.get(si) < 0) {}
                      termTuple = results(si)
                    }
                  }
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
                  val termTopicCounter = termTuple._2
                  val docTopicCounter = docTuple._2
                  val topics = data(i)
                  for (t <- topics) {
                    termTopicCounter.synchronized {
                      termTopicCounter(t) += 1
                    }
                    docTopicCounter.synchronized {
                      docTopicCounter(t) += 1
                    }
                  }
                }
              }
            } catch {
              case e: Exception => logger.error(e.getLocalizedMessage, e)
            } finally {
              doneSignal.countDown()
            }
          }
        }, s"aggregateLocal thread $threadId")
      }
      threads.foreach(_.start())
      doneSignal.await()
      results.filter(_ != null)
    })).partitionBy(vertices.partitioner.get)

    val partRDD = vertices.partitionsRDD.zipPartitions(shippedCounters, preservesPartitioning=true)(
      (svpIter, cntsIter) => svpIter.map(svp => {
        val results = svp.values
        val index = svp.index
        val marks = new AtomicIntegerArray(results.length)
        if (numThreads == 1) {
          cntsIter.foreach {
            case (vid, counter) =>
              val i = index.getPos(vid)
              if (marks.getAndDecrement(i) == 0) {
                results(i) = BSV.zeros[Count](numTopics)
              }
              results(i) :+= counter
              marks.set(i, Int.MaxValue)
          }
        } else {
          val numConsumers = numThreads - 1
          val queue = new ConcurrentLinkedQueue[(VertexId, TC)]()
          val doneSignal = new CountDownLatch(numConsumers)
          val threads = new Array[Thread](numConsumers)
          for (threadId <- threads.indices) {
            threads(threadId) = new Thread(new Runnable {
              override def run(): Unit = {
                val logger = Logger.getLogger(this.getClass.getName)
                var incomplete = true
                try {
                  while (incomplete) {
                    var t = queue.poll()
                    while (t == null) {
                      t = queue.poll()
                    }
                    val (vid, counter) = t
                    if (counter == null) {
                      incomplete = false
                    } else {
                      val i = index.getPos(vid)
                      if (marks.getAndDecrement(i) == 0) {
                        results(i) = BSV.zeros[Count](numTopics)
                        marks.set(i, Int.MaxValue)
                      } else {
                        while (marks.get(i) < 0) {}
                      }
                      val agg = results(i)
                      agg.synchronized {
                        agg :+= counter
                      }
                      marks.set(i, Int.MaxValue)
                    }
                  }
                } catch {
                  case e: Exception => logger.error(e.getLocalizedMessage, e)
                } finally {
                  doneSignal.countDown()
                }
              }
            }, s"aggregateGlobal thread $threadId")
          }
          threads.foreach(_.start())
          cntsIter.foreach(queue.offer)
          Range(0, numConsumers).foreach(thid => queue.offer((thid, null)))
          doneSignal.await()
        }
        svp.withValues(results)
      })
    )
    GraphImpl.fromExistingRDDs(vertices.withPartitionsRDD(partRDD), edges)
  }
}
