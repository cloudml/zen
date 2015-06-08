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

import org.apache.spark.{HashPartitioner, Partitioner}
import org.apache.spark.graphx._
import org.apache.spark.graphx.impl.GraphImpl
import org.apache.spark.storage.StorageLevel

import scala.math._
import scala.reflect.ClassTag

/**
 * Degree-Based Hashing, the paper:
 * Distributed Power-law Graph Computing: Theoretical and Empirical Analysis
 */
private[ml] class DBHPartitioner(val partitions: Int, val threshold: Int = 0) extends Partitioner {
  val mixingPrime: Long = 1125899906842597L

  override def numPartitions: Int = partitions

  /**
   * Default DBH doesn't consider the situation where both the degree of src and
   * dst vertices are both small than a given threshold value
   */
  def getPartition(key: Any): Int = {
    val edge = key.asInstanceOf[EdgeTriplet[Int, _]]
    val srcDeg = edge.srcAttr
    val dstDeg = edge.dstAttr
    val srcId = edge.srcId
    val dstId = edge.dstId
    val minId = if (srcDeg < dstDeg) srcId else dstId
    val maxId = if (srcDeg < dstDeg) dstId else srcId
    val maxDeg = if (srcDeg < dstDeg) dstDeg else srcDeg
    if (maxDeg < threshold) {
      getPartition(maxId)
    } else {
      getPartition(minId)
    }
  }

  def getPartition(idx: Int): PartitionID = {
    getPartition(idx.toLong)
  }

  def getPartition(idx: Long): PartitionID = {
    (abs(idx * mixingPrime) % partitions).toInt
  }

  override def equals(other: Any): Boolean = other match {
    case h: DBHPartitioner =>
      h.numPartitions == numPartitions
    case _ =>
      false
  }

  override def hashCode: Int = numPartitions
}

object DBHPartitioner {
  private[zen] def partitionByDBH[VD: ClassTag, ED: ClassTag](
    input: Graph[VD, ED],
    storageLevel: StorageLevel): Graph[VD, ED] = {
    val edges = input.edges
    val vertices = input.vertices
    val degrees = input.degrees.setName("degrees").persist(storageLevel)
    val degreesGraph = GraphImpl(degrees, edges)
    degreesGraph.persist(storageLevel)
    val numPartitions = edges.partitions.length
    val partitionStrategy = new DBHPartitioner(numPartitions, 0)
    val newEdges = degreesGraph.triplets.mapPartitions { iter =>
      iter.map { e =>
        (partitionStrategy.getPartition(e), Edge(e.srcId, e.dstId, e.attr))
      }
    }.partitionBy(new HashPartitioner(numPartitions)).map(_._2).persist(storageLevel)
    val dataSet = GraphImpl(vertices, newEdges, null.asInstanceOf[VD], storageLevel, storageLevel)
    dataSet.persist(storageLevel)
    dataSet.vertices.count()
    dataSet.edges.count()
    degrees.unpersist(blocking = false)
    degreesGraph.vertices.unpersist(blocking = false)
    degreesGraph.edges.unpersist(blocking = false)
    input.vertices.unpersist(blocking = false)
    input.edges.unpersist(blocking = false)
    newEdges.unpersist(blocking = false)
    dataSet
  }
}
