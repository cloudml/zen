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

package com.github.cloudml.zen.ml

import breeze.linalg.{SparseVector => BSV}
import org.apache.spark.Partitioner
import org.apache.spark.graphx._
import org.apache.spark.graphx.impl.GraphImpl
import org.apache.spark.storage.StorageLevel

/**
 * Bounded & Balanced Rearranger Partitioner
 */
private[ml] class BBRPartitioner(val partitions: Int) extends Partitioner {

  override def numPartitions: Int = partitions

  def getKey(et: EdgeTriplet[Int, _]): Long = {
    if (et.srcAttr >= et.dstAttr) et.srcId else et.dstId
  }

  def getPartition(key: Any): PartitionID = {
    (key.asInstanceOf[Long] % numPartitions).toInt
  }

  override def equals(other: Any): Boolean = other match {
    case tr: BBRPartitioner =>
      tr.numPartitions == numPartitions
    case _ =>
      false
  }

  override def hashCode: Int = numPartitions
}

object BBRPartitioner {
  private[zen] def partitionByBBR[VD, ED](input: Graph[VD, ED],
    numEdges: Long,
    storageLevel: StorageLevel): Graph[VD, ED] = {
    val edges = input.edges
    val vertices = input.vertices
    val numPartitions = edges.partitions.length
    val bbrp = new BBRPartitioner(numPartitions)
    val degGraph = GraphImpl(input.degrees, edges)
    val occurs = degGraph.triplets.mapPartitions(_.map(edge => (bbrp.getKey(edge), 1L)))
      .reduceByKey(_ + _).collect()
    val newEdges = input.triplets.mapPartitions { iter =>
      iter.map(e => (bbrp.getKey(e), Edge(e.srcId, e.dstId, e.attr)))
    }.partitionBy(bbrp).map(_._2)
    GraphImpl(vertices, newEdges, null.asInstanceOf[VD], storageLevel, storageLevel)
  }

  private def rearrage(occurs: Array[(VertexId, Long)], numEdges: Long, numPartitions: Int):Unit = {
    val numKeys = occurs.length
    val npp = numEdges / numPartitions
    val rpn = numEdges - npp * numPartitions
    val keyPartCount = new Array[(VertexId, BSV[Long])](numKeys)
    var pid = 0
    for (i <- 0 until numKeys) {
      val (vid, occur) = occurs(i)
      val
      val nrpp = npp + (if (pid < rpn) 1L else 0L)
      keyPartCount(i) = (vid, )
    }
  }
}
