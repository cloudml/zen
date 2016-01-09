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
import scala.language.existentials
import scala.reflect.ClassTag

import com.github.cloudml.zen.ml.sampler._

import breeze.collection.mutable.SparseArray
import breeze.linalg.{Vector => BV, SparseVector => BSV, DenseVector => BDV}
import breeze.storage.Zero
import org.apache.spark.SparkConf
import org.apache.spark.graphx2._
import org.apache.spark.graphx2.impl._


object LDADefines {
  type DocId = VertexId
  type WordId = VertexId
  type Count = Int
  type TC = Product
  type TA = Int
  type BOW = (Long, BSV[Count])

  val sv_formatVersionV2_0 = "2.0"
  val sv_classNameV2_0 = "com.github.cloudml.zen.ml.clustering.DistributedLDAModel"
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
  val cs_ignoreDocId = "zen.lda.ignoreDocId"
  val cs_saveTransposed = "zen.lda.saveTransposed"

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
    var i = 0
    while (i < tokens.length) {
      val topic = gen.nextInt(numTopics)
      topics(i) = topic
      docTopics(topic) += 1
      i += 1
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
    val used = bsv.used
    val index = bsv.index
    val data = bsv.data
    var i = 0
    while (i < used) {
      bdv(index(i)) = data(i)
      i += 1
    }
    bdv
  }

  def refreshEdgeAssociations[VD: ClassTag, ED: ClassTag](graph: Graph[VD, ED]): GraphImpl[VD, ED] = {
    val gimpl = graph.asInstanceOf[GraphImpl[VD, ED]]
    val vertices = gimpl.vertices
    val edges = gimpl.edges.asInstanceOf[EdgeRDDImpl[ED, VDO] forSome { type VDO }]
    val numThreads = edges.context.getConf.getInt(cs_numThreads, 1)
    val shippedVerts = vertices.partitionsRDD.mapPartitions(_.flatMap(svp => {
      val rt = svp.routingTable
      val index = svp.index
      val values = svp.values
      implicit val es = ExecutionContext.fromExecutorService(Executors.newFixedThreadPool(numThreads))
      Range(0, rt.numEdgePartitions).grouped(numThreads).flatMap(batch => {
        val all = Future.traverse(batch)(pid => Future {
          val vids = rt.routingTable(pid)._1
          val attrs = vids.map(vid => values(index.getPos(vid)))
          (pid, new VertexAttributeBlock(vids, attrs))
        })
        Await.result(all, 1.hour)
      }) ++ {
        es.shutdown()
        Iterator.empty
      }
    })).partitionBy(edges.partitioner.get)

    // Below identical map is used to isolate the impact of locality of CheckpointRDD
    val isoRDD = edges.partitionsRDD.mapPartitions(iter => iter, preservesPartitioning=true)
    val partRDD = isoRDD.zipPartitions(shippedVerts, preservesPartitioning=true)(
      (epIter, vabsIter) => epIter.map(Function.tupled((pid, ep) => {
        val g2l = ep.global2local
        val results = new Array[VD](ep.vertexAttrs.length)
        implicit val es = ExecutionContext.fromExecutorService(Executors.newFixedThreadPool(numThreads))
        val all = Future.traverse(vabsIter)(Function.tupled((_, vab) => Future {
          vab.iterator.foreach(Function.tupled((vid, vdata) =>
            results(g2l(vid)) = vdata
          ))
        }))
        Await.ready(all, 1.hour)
        es.shutdown()
        (pid, ep.withVertexAttributes(results))
      }))
    )
    GraphImpl.fromExistingRDDs(vertices, edges.withPartitionsRDD(partRDD))
  }
}
