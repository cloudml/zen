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
import scala.concurrent.duration.Duration
import scala.reflect.ClassTag

import com.github.cloudml.zen.ml.util._

import breeze.collection.mutable.SparseArray
import breeze.linalg.{Vector => BV, SparseVector => BSV, DenseVector => BDV}
import breeze.storage.Zero
import org.apache.spark.SparkConf
import org.apache.spark.graphx2._
import org.apache.spark.graphx2.impl._
import org.apache.spark.graphx2.util.collection.GraphXPrimitiveVector


object LDADefines {
  type DocId = VertexId
  type WordId = VertexId
  type Count = Int
  type TC = BV[Count]
  type TA = Array[Int]
  type BOW = (Long, BSV[Count])

  val sv_formatVersionV1_0 = "1.0"
  val sv_classNameV1_0 = "com.github.cloudml.zen.ml.clustering.DistributedLDAModel"
  val cs_numTopics = "zen.lda.numTopics"
  val cs_numPartitions = "zen.lda.numPartitions"
  val cs_sampleRate = "zen.lda.sampleRate"
  val cs_LDAAlgorithm = "zen.lda.LDAAlgorithm"
  val cs_accelMethod = "zen.lda.accelMethod"
  val cs_storageLevel = "zen.lda.storageLevel"
  val cs_partStrategy = "zen.lda.partStrategy"
  val cs_initStrategy = "zen.lda.initStrategy"
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
    val docTopics = BSV.zeros[Count](numTopics)
    for (i <- tokens.indices) {
      val topic = gen.nextInt(numTopics)
      topics(i) = topic
      docTopics(topic) += 1
    }
    docTopics
  }

  def registerKryoClasses(conf: SparkConf): Unit = {
    conf.registerKryoClasses(Array(
      classOf[TC], classOf[TA],
      classOf[BOW],
      classOf[(TC, Double, Int)],  // for perplexity
      classOf[AliasTable[Object]], classOf[FTree[Object]],  // for some partitioners
      classOf[BSV[Object]], classOf[BDV[Object]],
      classOf[SparseArray[Object]]  // member of BSV
    ))
  }

  def toBDV[@specialized(Int, Double, Float) V: ClassTag: Zero](bsv: BSV[V]): BDV[V] = {
    val bdv = BDV.zeros[V](bsv.length)
    for ((i, v) <- bsv.activeIterator) {
      bdv(i) = v
    }
    bdv
  }

  def refreshEdgeAssociations[VD: ClassTag, ED: ClassTag](graph: Graph[VD, ED]): GraphImpl[VD, ED] = {
    val gimpl = graph.asInstanceOf[GraphImpl[VD, ED]]
    val vertices = gimpl.vertices
    val edges = gimpl.edges.asInstanceOf[EdgeRDDImpl[ED, _]]
    val numThreads = edges.context.getConf.getInt(cs_numThreads, 1)
    val shippedVerts = vertices.partitionsRDD.mapPartitions(_.flatMap(svp => {
      val rt = svp.routingTable
      implicit val ec = ExecutionContext.fromExecutorService(Executors.newFixedThreadPool(numThreads))
      Range(0, rt.numEdgePartitions).grouped(numThreads).flatMap(batch => {
        val all = Future.traverse(batch)(pid => Future {
          val totalSize = rt.partitionSize(pid)
          val vids = new GraphXPrimitiveVector[VertexId](totalSize)
          val attrs = new GraphXPrimitiveVector[VD](totalSize)
          rt.foreachWithinEdgePartition(pid, includeSrc = true, includeDst = true)(vid => {
            if (svp.isDefined(vid)) {
              vids += vid
              attrs += svp(vid)
            }
          })
          (pid, new VertexAttributeBlock(vids.trim().array, attrs.trim().array))
        })
        Await.result(all, Duration.Inf)
      }) ++ {
        ec.shutdown()
        Iterator.empty
      }
    })).partitionBy(edges.partitioner.get)

    val partRDD = edges.partitionsRDD.zipPartitions(shippedVerts, preservesPartitioning=true)(
      (epIter, vabsIter) => epIter.map {
        case (pid, ep) =>
          val g2l = ep.global2local
          val results = new Array[VD](ep.vertexAttrs.length)
          implicit val ec = ExecutionContext.fromExecutorService(Executors.newFixedThreadPool(numThreads))
          val all = Future.traverse(vabsIter)(t => Future {
            val vab = t._2
            for ((vid, vdata) <- vab.iterator) {
              results(g2l(vid)) = vdata
            }
          })
          Await.ready(all, Duration.Inf)
          ec.shutdown()
          (pid, ep.withVertexAttributes(results))
      }
    )
    GraphImpl.fromExistingRDDs(vertices, edges.withPartitionsRDD(partRDD))
  }
}
