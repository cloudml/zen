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

package com.github.cloudml.zen.ml.partitioner

import scala.reflect.ClassTag

import com.github.cloudml.zen.ml.clustering.LDADefines._
import com.github.cloudml.zen.ml.sampler.AliasTable
import com.github.cloudml.zen.ml.util.XORShiftRandom
import breeze.linalg.{SparseVector => BSV}
import org.apache.spark.Partitioner
import org.apache.spark.graphx2._
import org.apache.spark.graphx2.impl.GraphImpl
import org.apache.spark.storage.StorageLevel


private[ml] class BBRPartitioner(val partitions: Int) extends Partitioner {

  override def numPartitions: Int = partitions

  def getKey(et: EdgeTriplet[Int, _]): VertexId = {
    if (et.srcAttr >= et.dstAttr) et.srcId else et.dstId
  }

  def getPartition(key: Any): PartitionID = {
    key.asInstanceOf[PartitionID] % numPartitions
  }

  override def equals(other: Any): Boolean = other match {
    case bbr: BBRPartitioner =>
      bbr.numPartitions == numPartitions
    case _ =>
      false
  }

  override def hashCode: Int = numPartitions
}

/**
 * Bounded & Balanced Rearranger Partitioner
 */
object BBRPartitioner {
  private[zen] def partitionByBBR[VD: ClassTag, ED: ClassTag](
    input: Graph[VD, ED],
    storageLevel: StorageLevel): Graph[VD, ED] = {
    val edges = input.edges
    val conf = edges.context.getConf
    val numPartitions = conf.getInt(cs_numPartitions, edges.partitions.length)
    val bbr = new BBRPartitioner(numPartitions)
    val degGraph = GraphImpl(input.degrees, edges)
    val assnGraph = degGraph.mapTriplets((pid, iter) =>
      iter.map(et => (bbr.getKey(et), Edge(et.srcId, et.dstId, et.attr))), TripletFields.All)
    assnGraph.persist(storageLevel)

    val assnVerts = assnGraph.aggregateMessages[Long](ect => {
      if (ect.attr._1 == ect.srcId) {
        ect.sendToSrc(1L)
      } else {
        ect.sendToDst(1L)
      }
    }, _ + _, TripletFields.EdgeOnly)
    val (kids, koccurs) = assnVerts.filter(_._2 > 0L).collect().unzip
    val partRdd = edges.context.parallelize(kids.zip(rearrage(koccurs, numPartitions)))
    val rearrGraph = assnGraph.mapVertices((_, _) => null.asInstanceOf[AliasTable[Long]])
      .joinVertices(partRdd)((_, _, arr) => AliasTable.generateAlias(arr))

    val newEdges = rearrGraph.triplets.mapPartitions(iter => {
      val gen = new XORShiftRandom()
      iter.map(et => {
        val (kid, edge) = et.attr
        val table = if (kid == et.srcId) et.srcAttr else et.dstAttr
        (table.sampleRandom(gen), edge)
      })
    }).partitionBy(bbr).map(_._2)
    GraphImpl(input.vertices, newEdges, null.asInstanceOf[VD], storageLevel, storageLevel)
  }

  private def rearrage(koccurs: IndexedSeq[Long], numPartitions: Int): IndexedSeq[BSV[Long]] = {
    val numKeys = koccurs.length
    val numEdges = koccurs.sum
    val npp = numEdges / numPartitions
    val rpn = numEdges - npp * numPartitions
    @inline def nrpp(pi: Int): Long = npp + (if (pi < rpn) 1L else 0L)
    @inline def kbn(ki: Int): Long = if (ki < numKeys) koccurs(ki) else 0L
    val keyPartCount = koccurs.map(t => BSV.zeros[Long](numPartitions))
    def put(ki: Int, krest: Long, pi: Int, prest: Long): Unit = {
      if (ki < numKeys) {
        if (krest == prest) {
          keyPartCount(ki)(pi) = krest
          put(ki + 1, kbn(ki + 1), pi + 1, nrpp(pi + 1))
        } else if (krest < prest) {
          keyPartCount(ki)(pi) = krest
          put(ki + 1, kbn(ki + 1), pi, prest - krest)
        } else {
          keyPartCount(ki)(pi) = prest
          put(ki, krest - prest, pi + 1, nrpp(pi + 1))
        }
      }
    }
    put(0, kbn(0), 0, nrpp(0))
    keyPartCount
  }
}
