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

import org.apache.spark.Partitioner
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
  def getKey(et: EdgeTriplet[Int, _]): Long = {
    val srcId = et.srcId
    val dstId = et.dstId
    val srcDeg = et.srcAttr
    val dstDeg = et.dstAttr
    val minId = min(srcId, dstId)
    val maxId = max(srcId, dstId)
    val maxDeg = max(srcDeg, dstDeg)
    if (maxDeg < threshold) {
      maxId
    } else {
      minId
    }
  }

  def getPartition(key: Any): PartitionID = {
    val idx = key.asInstanceOf[Long]
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
    val numPartitions = edges.partitions.length
    val dbh = new DBHPartitioner(numPartitions, 0)
    val degGraph = GraphImpl(input.degrees, edges)
    val newEdges = degGraph.triplets.mapPartitions { iter =>
      iter.map(e => (dbh.getKey(e), Edge(e.srcId, e.dstId, e.attr)))
    }.partitionBy(dbh).map(_._2)
    GraphImpl(vertices, newEdges, null.asInstanceOf[VD], storageLevel, storageLevel)
  }
}
