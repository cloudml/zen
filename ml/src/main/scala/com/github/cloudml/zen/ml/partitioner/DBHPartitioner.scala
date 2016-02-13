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

import org.apache.spark.HashPartitioner
import org.apache.spark.graphx2._
import org.apache.spark.graphx2.impl.GraphImpl
import org.apache.spark.storage.StorageLevel

/**
 * Degree-Based Hashing, the paper:
 * Distributed Power-law Graph Computing: Theoretical and Empirical Analysis
 */
class DBHPartitioner(val partitions: Int, val threshold: Int = 0)
  extends HashPartitioner(partitions) {
  /**
   * Default DBH doesn't consider the situation where both the degree of src and
   * dst vertices are both small than a given threshold value
   */
  def getKey(et: EdgeTriplet[Int, _]): Long = {
    val srcId = et.srcId
    val dstId = et.dstId
    val srcDeg = et.srcAttr
    val dstDeg = et.dstAttr
    val maxDeg = math.max(srcDeg, dstDeg)
    val minDegId = if (maxDeg == srcDeg) dstId else srcId
    val maxDegId = if (maxDeg == srcDeg) srcId else dstId
    if (maxDeg < threshold) {
      maxDegId
    } else {
      minDegId
    }
  }

  override def equals(other: Any): Boolean = other match {
    case dbh: DBHPartitioner =>
      dbh.numPartitions == numPartitions
    case _ =>
      false
  }
}

object DBHPartitioner {
  def partitionByDBH[VD: ClassTag, ED: ClassTag](input: Graph[VD, ED],
    storageLevel: StorageLevel): Graph[VD, ED] = {
    val edges = input.edges
    val conf = edges.context.getConf
    val numPartitions = conf.getInt(cs_numPartitions, edges.partitions.length)
    val dbh = new DBHPartitioner(numPartitions, 0)
    val degGraph = GraphImpl(input.degrees, edges)
    val newEdges = degGraph.triplets.mapPartitions(_.map(et =>
      (dbh.getKey(et), Edge(et.srcId, et.dstId, et.attr))
    )).partitionBy(dbh).map(_._2)
    GraphImpl(input.vertices, newEdges, null.asInstanceOf[VD], storageLevel, storageLevel)
  }
}
