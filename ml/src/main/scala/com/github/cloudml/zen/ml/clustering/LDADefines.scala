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

import breeze.collection.mutable.SparseArray
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}
import breeze.storage.Zero
import com.github.cloudml.zen.ml.sampler._
import com.github.cloudml.zen.ml.util.CompressedVector
import org.apache.spark.SparkConf
import org.apache.spark.graphx2._

import scala.reflect.ClassTag


object LDADefines {
  type DocId = VertexId
  type WordId = VertexId
  type Count = Int
  type TC = CompressedVector
  type TA = Int
  type BOW = (Long, BSV[Count])
  type Nwk = BV[Count]
  type Ndk = BSV[Count]
  type Nvk = BV[Count]
  type NwkPair = (VertexId, Nwk)
  type NvkPair = (VertexId, Nvk)

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
      classOf[Tuple1[Object]], classOf[(Object, Object)],
      classOf[BOW],
      classOf[NwkPair],
      classOf[(TC, Double, Int)],  // for perplexity
      classOf[AliasTable[Object]], classOf[FTree[Object]],  // for some partitioners
      classOf[BSV[Object]], classOf[BDV[Object]],
      classOf[SparseArray[Object]],  // member of BSV
      classOf[Array[Int]]
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
}
