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

import com.github.cloudml.zen.ml.util.{AliasTable, XORShiftRandom}
import breeze.linalg.{SparseVector => BSV}
import org.apache.spark.Partitioner
import org.apache.spark.graphx._
import org.apache.spark.graphx.impl.GraphImpl
import org.apache.spark.storage.StorageLevel


private[ml] class BBRPartitioner(val partitions: Int) extends Partitioner {

  override def numPartitions: Int = partitions

  def getKey(et: EdgeTriplet[VertexId, _]): VertexId = {
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
    val numPartitions = input.edges.partitions.length
    val bbr = new BBRPartitioner(numPartitions)
    val degGraph = GraphImpl(input.degrees.mapValues(_.toLong), input.edges)
    degGraph.persist(storageLevel)

    val docOccurs = assignVerts(assignGraph(degGraph, bbr)).filter(_._1 < 0L)
    val adjGraph = degGraph.joinVertices(docOccurs)((_, _, docOcc) => docOcc)
    val assnGraph = assignGraph(adjGraph, bbr)
    assnGraph.persist(storageLevel)
    val (kids, koccurs) = assignVerts(assnGraph).filter(_._2 > 0L).collect().unzip
    val partRdd = input.edges.context.parallelize(kids.zip(rearrage(koccurs, numPartitions)))
    val rearrGraph = assnGraph.mapVertices((_, _) => null.asInstanceOf[AliasTable[Long]])
      .joinVertices(partRdd)((_, _, arr) => AliasTable.generateAlias(arr))

    val pidGraph = rearrGraph.mapTriplets((pid, iter) => {
      val gen = new XORShiftRandom()
      iter.map(et => {
        val table = if (et.attr == et.srcId) et.srcAttr else et.dstAttr
        table.sampleRandom(gen)
      })
    }, TripletFields.All)
    val newEdges = pidGraph.edges.innerJoin(degGraph.edges)((_, _, pid, data) => (pid, data))
      .mapPartitions(_.map(e =>
      (e.attr._1, Edge(e.srcId, e.dstId, e.attr._2))), preservesPartitioning=true)
      .partitionBy(bbr).map(_._2)
    GraphImpl(input.vertices, newEdges, null.asInstanceOf[VD], storageLevel, storageLevel)
  }

  private def assignGraph[ED](cntGraph: Graph[Long, _], bbr: BBRPartitioner): Graph[Long, VertexId] = {
    cntGraph.mapTriplets((pid, iter) => iter.map(bbr.getKey), TripletFields.All)
  }

  private def assignVerts(assnGraph: Graph[_, VertexId]): VertexRDD[Long] = {
    assnGraph.aggregateMessages[Long](ect => {
      if (ect.attr == ect.srcId) {
        ect.sendToSrc(1L)
      } else {
        ect.sendToDst(1L)
      }
    }, _ + _, TripletFields.All)
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
