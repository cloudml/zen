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
import scala.reflect.ClassTag

import com.github.cloudml.zen.ml.partitioner._
import com.github.cloudml.zen.ml.util.{FTree, AliasTable, XORShiftRandom}
import breeze.linalg.{SparseVector => BSV, DenseVector => BDV}
import org.apache.log4j.Logger
import org.apache.spark.SparkConf
import org.apache.spark.graphx2._
import org.apache.spark.graphx2.impl._
import org.apache.spark.graphx2.util.collection.GraphXPrimitiveVector


object LDADefines {
  type DocId = VertexId
  type WordId = VertexId
  type Count = Int
  type TC = BSV[Count]
  type TA = Array[Int]
  type BOW = (Long, BSV[Count])

  val sv_formatVersionV1_0 = "1.0"
  val sv_classNameV1_0 = "com.github.cloudml.zen.ml.clustering.DistributedLDAModel"
  val cs_sampleRate = "zen.lda.sampleRate"
  val cs_LDAAlgorithm = "zen.lda.LDAAlgorithm"
  val cs_accelMethod = "zen.lda.accelMethod"
  val cs_storageLevel = "zen.lda.storageLevel"
  val cs_partStrategy = "zen.lda.partStrategy"
  val cs_chkptInterval = "zen.lda.chkptInterval"
  val cs_calcPerplexity = "zen.lda.calcPerplexity"
  val cs_saveInterval = "zen.lda.saveInterval"
  val cs_inputPath = "zen.lda.inputPath"
  val cs_outputpath = "zen.lda.outputPath"
  val cs_saveAsSolid = "zen.lda.saveAsSolid"
  val cs_numThreads = "zen.lda.numThreads"

  // make docId always be negative, so that the doc vertex always be the dest vertex
  @inline def genNewDocId(docId: Long): VertexId = {
    assert(docId >= 0)
    -(docId + 1L)
  }

  @inline def isDocId(vid: VertexId): Boolean = vid < 0L

  @inline def isTermId(vid: VertexId): Boolean = vid >= 0L

  def uniformDistSampler(gen: Random,
    tokens: Array[Int],
    topics: Array[Int],
    numTopics: Int): BSV[Count] = {
    val docTopicCounter = BSV.zeros[Count](numTopics)
    for (i <- tokens.indices) {
      val topic = gen.nextInt(numTopics)
      topics(i) = topic
      docTopicCounter(topic) += 1
    }
    docTopicCounter
  }

  def registerKryoClasses(conf: SparkConf): Unit = {
    conf.registerKryoClasses(Array(classOf[TC], classOf[TA],
      classOf[BSV[Double]], classOf[BDV[Count]], classOf[BDV[Double]],
      classOf[BOW], classOf[Random], classOf[XORShiftRandom],
      classOf[LDA], classOf[LocalLDAModel], classOf[DistributedLDAModel],
      classOf[LDAAlgorithm], classOf[FastLDA], classOf[LightLDA],
      classOf[DBHPartitioner], classOf[VSDLPPartitioner], classOf[BBRPartitioner],
      classOf[AliasTable[Double]], classOf[AliasTable[Object]], classOf[FTree[Double]], classOf[FTree[Object]]
    ))
  }

  def refreshEdgeAssociations[VD: ClassTag, ED: ClassTag](graph: Graph[VD, ED]): GraphImpl[VD, ED] = {
    val gimpl = graph.asInstanceOf[GraphImpl[VD, ED]]
    val vertices = gimpl.vertices
    val edges = gimpl.edges
    val numThreads = edges.context.getConf.getInt(cs_numThreads, 1)
    val shippedVerts = vertices.partitionsRDD.mapPartitions(_.flatMap(svp => {
      val rt = svp.routingTable
      val totalSize = rt.numEdgePartitions
      val results = new Array[(PartitionID, VertexAttributeBlock[VD])](totalSize)
      val sizePerThrd = {
        val npt = totalSize / numThreads
        if (npt * numThreads == totalSize) npt else npt + 1
      }
      val doneSignal = new CountDownLatch(numThreads)
      val threads = new Array[Thread](numThreads)
      for (threadId <- threads.indices) {
        threads(threadId) = new Thread(new Runnable {
          val startPos = sizePerThrd * threadId
          val endPos = math.min(sizePerThrd * (threadId + 1), totalSize)

          override def run(): Unit = {
            val logger = Logger.getLogger(this.getClass.getName)
            try {
              for (i <- startPos until endPos) {
                val initialSize = rt.partitionSize(i)
                val vids = new GraphXPrimitiveVector[VertexId](initialSize)
                val attrs = new GraphXPrimitiveVector[VD](initialSize)
                rt.foreachWithinEdgePartition(i, includeSrc = true, includeDst = true)(vid => {
                  if (svp.isDefined(vid)) {
                    vids += vid
                    attrs += svp(vid)
                  }
                })
                val vab = new VertexAttributeBlock(vids.trim().array, attrs.trim().array)
                results(i) = (i, vab)
              }
            } catch {
              case e: Exception => logger.error(e.getLocalizedMessage, e)
            } finally {
              doneSignal.countDown()
            }
          }
        }, s"shipVertAttrs thread $threadId")
      }
      threads.foreach(_.start())
      doneSignal.await()
      results
    })).partitionBy(edges.partitioner.get)

    val partRDD = edges.partitionsRDD.zipPartitions(shippedVerts, preservesPartitioning=true)(
      (epIter, vabsIter) => epIter.map {
        case (pid, ep) =>
          val vattrs = ep.vertexAttrs
          val g2l = ep.global2local
          val results = new Array[VD](vattrs.length)
          val doneSignal = new CountDownLatch(numThreads)
          val threads = new Array[Thread](numThreads)
          for (threadId <- threads.indices) {
            threads(threadId) = new Thread(new Runnable {
              override def run(): Unit = {
                val logger = Logger.getLogger(this.getClass.getName)
                try {
                  var t: (PartitionID, VertexAttributeBlock[VD]) = null
                  var isOver = false
                  while (!isOver) {
                    vabsIter.synchronized {
                      t = if (vabsIter.hasNext) {
                        vabsIter.next()
                      } else {
                        isOver = true
                        null
                      }
                    }
                    if (t != null) {
                      for ((vid, vdata) <- t._2.iterator) {
                        results(g2l(vid)) = vdata
                      }
                    }
                  }
                } catch {
                  case e: Exception => logger.error(e.getLocalizedMessage, e)
                } finally {
                  doneSignal.countDown()
                }
              }
            }, s"refreshEdges thread $threadId")
          }
          threads.foreach(_.start())
          doneSignal.await()
          val newEp = new EdgePartition(ep.localSrcIds, ep.localDstIds, ep.data, ep.index, g2l,
            ep.local2global, results, ep.activeSet)
          (pid, newEp)
      }
    )
    GraphImpl.fromExistingRDDs(vertices, edges.withPartitionsRDD(partRDD))
  }
}
