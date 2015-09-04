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
import java.util.concurrent.CountDownLatch

import LDA._
import LDADefines._
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, sum => brzSum}
import com.github.cloudml.zen.ml.partitioner._
import com.github.cloudml.zen.ml.util.XORShiftRandom
import org.apache.log4j.Logger
import org.apache.spark.graphx2._
import org.apache.spark.graphx2.impl.GraphImpl
import org.apache.spark.mllib.linalg.{SparseVector => SSV, Vector => SV}
import org.apache.spark.mllib.linalg.distributed.{MatrixEntry, RowMatrix}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel


class LDA(@transient private var corpus: Graph[VD, ED],
  private val numTopics: Int,
  private val numTerms: Int,
  private val numDocs: Long,
  private val numTokens: Long,
  private var alpha: Double,
  private var beta: Double,
  private var alphaAS: Double,
  private val algo: LDAAlgorithm,
  private var storageLevel: StorageLevel) extends Serializable {

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
      if (pplx) {
        println(s"Gibbs sampling (Iteration $iter/$totalIter): perplexity=${perplexity()}")
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
    sampledCorpus.persist(storageLevel)
    sampledCorpus.edges.setName(s"sampledEdges-$sampIter")
    sampledCorpus.vertices.setName(s"sampledVertices-$sampIter")

    corpus = if (inferenceOnly) {
      updateDocTopicCounter(sampledCorpus, numTopics)
    } else {
      updateCounter(sampledCorpus, numTopics)
    }
    if (chkptIntv > 0 && sampIter % chkptIntv == 1 && sc.getCheckpointDir.isDefined) {
      corpus.checkpoint()
    }
    corpus.persist(storageLevel)
    corpus.edges.setName(s"edges-$sampIter").count()
    corpus.vertices.setName(s"vertices-$sampIter")
    totalTopicCounter = collectTopicCounter()

    prevCorpus.unpersist(blocking=false)
    sampledCorpus.unpersist(blocking=false)
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
    val tCounter = this.totalTopicCounter
    val numTokens = this.numTokens
    val alphaAS = this.alphaAS
    val alphaSum = this.numTopics * this.alpha
    val alphaRatio = alphaSum / (numTokens + alphaAS * this.numTopics)
    val beta = this.beta
    val betaSum = this.numTerms * beta

    // \frac{{\alpha }_{k}{\beta }_{w}}{{n}_{k}+\bar{\beta }}
    val tDenseSum = tCounter.valuesIterator.map(c => beta * alphaRatio * (c + alphaAS) / (c + betaSum)).sum

    val termProb = corpus.mapVertices((vid, counter) => {
      val probDist = if (isDocId(vid)) {
        counter.mapActivePairs((t, c) => c * beta / (tCounter(t) + betaSum))
      } else {
        counter.mapActivePairs((t, c) => c * alphaRatio * (tCounter(t) + alphaAS) / (tCounter(t) + betaSum))
      }
      val cSum = if (isDocId(vid)) brzSum(counter) else 0
      (counter, brzSum(probDist), cSum)
    }).mapTriplets(triplet => {
      val (termTopicCounter, wSparseSum, _) = triplet.srcAttr
      val (docTopicCounter, dSparseSum, docSize) = triplet.dstAttr
      val occurs = triplet.attr.length

      // \frac{{n}_{kw}{n}_{kd}}{{n}_{k}+\bar{\beta}}
      val dwSparseSum = docTopicCounter.activeIterator.map { case (t, c) =>
        c * termTopicCounter(t) / (tCounter(t) + betaSum)
      }.sum
      val prob = (tDenseSum + wSparseSum + dSparseSum + dwSparseSum) / (docSize + alphaSum)

      Math.log(prob) * occurs
    }).edges.map(_.attr).sum()

    math.exp(-1 * termProb / numTokens)
  }
}

object LDA {
  def apply(bowDocs: RDD[BOW],
    numTopics: Int,
    alpha: Double,
    beta: Double,
    alphaAS: Double,
    algo: LDAAlgorithm,
    storageLevel: StorageLevel): LDA = {
    val numTerms = bowDocs.first()._2.size
    val numDocs = bowDocs.count()
    val corpus = initializeCorpus(bowDocs, numTopics, storageLevel)
    val numTokens = corpus.edges.map(e => e.attr.length.toLong).reduce(_ + _)
    new LDA(corpus, numTopics, numTerms, numDocs, numTokens, alpha, beta, alphaAS, algo, storageLevel)
  }

  // initialize LDA for inference or incremental training
  def apply(computedModel: DistributedLDAModel,
    bowDocs: RDD[BOW],
    algo: LDAAlgorithm): LDA = {
    val numTopics = computedModel.numTopics
    val numTerms = computedModel.numTerms
    val numTokens = computedModel.numTokens
    val alpha = computedModel.alpha
    val beta = computedModel.beta
    val alphaAS = computedModel.alphaAS
    val storageLevel = computedModel.storageLevel
    val corpus = initializeCorpus(bowDocs, numTopics, storageLevel)
    corpus.joinVertices(computedModel.termTopicCounters)((_, _, computedCounter) => computedCounter)
    val numDocs = bowDocs.count()
    new LDA(corpus, numTopics, numTerms, numDocs, numTokens, alpha, beta, alphaAS, algo, storageLevel)
  }

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
  def train(docs: RDD[BOW],
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

  def incrementalTrain(docs: RDD[BOW],
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

  private[ml] def initializeCorpus(
    docs: RDD[BOW],
    numTopics: Int,
    storageLevel: StorageLevel): Graph[VD, ED] = {
    val conf = docs.context.getConf
    val edges = docs.mapPartitionsWithIndex((pid, iter) => {
      val gen = new XORShiftRandom(pid + 117)
      iter.flatMap { case (docId, doc) =>
        initializeEdges(gen, doc, docId, numTopics)
      }
    })
    val initCorpus: Graph[VD, ED] = Graph.fromEdges(edges, null, storageLevel, storageLevel)
    initCorpus.persist(storageLevel)
    initCorpus.vertices.setName("initVertices")
    initCorpus.edges.setName("initEdges")
    val partCorpus = conf.get(cs_partStrategy, "dbh") match {
      case "edge2d" =>
        println("using Edge2D partition strategy.")
        initCorpus.partitionBy(PartitionStrategy.EdgePartition2D)
      case "dbh" =>
        println("using Degree-based Hashing partition strategy.")
        DBHPartitioner.partitionByDBH[VD, ED](initCorpus, storageLevel)
      case "vsdlp" =>
        println("using Vertex-cut Stochastic Dynamic Label Propagation partition strategy.")
        VSDLPPartitioner.partitionByVSDLP[VD, ED](initCorpus, 4, storageLevel)
      case "bbr" =>
        println("using Bounded & Balanced Rearranger partition strategy.")
        BBRPartitioner.partitionByBBR[VD, ED](initCorpus, storageLevel)
      case _ =>
        throw new NoSuchMethodException("No this algorithm or not implemented.")
    }
    partCorpus.persist(storageLevel)
    val corpus = updateCounter(partCorpus, numTopics)
    corpus.persist(storageLevel)
    corpus.vertices.setName("vertices-0").count()
    val numEdges = corpus.edges.setName("edges-0").count()
    println(s"edges in the corpus: $numEdges")
    docs.unpersist(blocking=false)
    initCorpus.unpersist(blocking=false)
    partCorpus.unpersist(blocking=false)
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

  // make docId always be negative, so that the doc vertex always be the dest vertex
  @inline private def genNewDocId(docId: Long): Long = {
    assert(docId >= 0)
    -(docId + 1L)
  }

  @inline private[ml] def isDocId(id: Long): Boolean = id < 0L

  @inline private[ml] def isTermId(id: Long): Boolean = id >= 0L

  private def updateCounter(corpus: Graph[VD, ED],
    numTopics: Int): Graph[VD, ED] = {
    val conf = corpus.edges.context.getConf
    val numThreads = conf.getInt(cs_numThreads, 1)
    val graph = corpus.asInstanceOf[GraphImpl[VD, ED]]
    val vertices = graph.vertices
    val edges = graph.replicatedVertexView.edges
    val newCounterPartition = edges.partitionsRDD.mapPartitions(_.flatMap(t => {
      val ep = t._2
      val totalSize = ep.size
      val verts = ep.vertexAttrs.map(t => BSV.zeros[Count](numTopics))
      val sizePerThrd = {
        val npt = totalSize / numThreads
        if (npt * numThreads == totalSize) npt else npt + 1
      }
      val doneSignal = new CountDownLatch(numThreads)
      val threads = new Array[Thread](numThreads)
      for (threadId <- threads.indices) {
        threads(threadId) = new Thread(new Runnable {
          val logger: Logger = Logger.getLogger(this.getClass.getName)
          val startPos = sizePerThrd * threadId
          val endPos = math.min(sizePerThrd * (threadId + 1), totalSize)

          override def run(): Unit = {
            try {
              for (i <- startPos until endPos) {
                val localSrcId = ep.localSrcIds(i)
                val localDstId = ep.localDstIds(i)
                val termTopicCounter = verts(localSrcId)
                val docTopicCounter = verts(localDstId)
                val topics = ep.data(i)
                for (t <- topics) {
                  termTopicCounter.synchronized { termTopicCounter(t) += 1 }
                  docTopicCounter.synchronized { docTopicCounter(t) += 1 }
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

      verts.zipWithIndex.map {
        case (cnts, i) => (ep.local2global(i), cnts)
      }
    })).partitionBy(vertices.partitioner.get)
    val newCounter = vertices.aggregateUsingIndex[VD](newCounterPartition, _ :+= _)

    val newVerts = vertices.leftJoin(newCounter)((id: VertexId, data: VD, o: Option[VD]) => {
      o match {
        case Some(u) => u
        case None => data
      }
    })
    val changedVerts = vertices.diff(newVerts)
    val shippedVerts = changedVerts.shipVertexAttributes(shipSrc=true, shipDst=true).partitionBy(edges.partitioner.get)
    val partRdd = edges.partitionsRDD.zipPartitions(shippedVerts, preservesPartitioning=true)((epIter, vabsIter) =>
      epIter.map { case (pid, edgePartition) =>
        (pid, edgePartition.updateVertices(vabsIter.flatMap(_._2.iterator)))
      }
    )
    val newEdges = edges.withPartitionsRDD(partRdd)
    GraphImpl.fromExistingRDDs(newVerts, newEdges)
  }

  private def updateDocTopicCounter(corpus: Graph[VD, ED],
    numTopics: Int): Graph[VD, ED] = {
    val newCounter = corpus.edges.mapPartitions(_.flatMap(edge =>
      Iterator.single((edge.dstId, edge.attr))
    )).aggregateByKey(BSV.zeros[Count](numTopics), corpus.vertices.partitioner.get)((agg, cur) => {
      cur.foreach(agg(_) += 1)
      agg
    }, _ :+= _)
    corpus.joinVertices(newCounter)((_, _, counter) => counter)
  }
}
