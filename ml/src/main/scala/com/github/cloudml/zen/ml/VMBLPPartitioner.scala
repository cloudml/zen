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

import breeze.linalg.{SparseVector => BSV, DenseMatrix}
import scala.reflect.ClassTag
import com.github.cloudml.zen.ml.util.{XORShiftRandom, AliasTable}
import org.apache.spark.Partitioner
import org.apache.spark.graphx._
import org.apache.spark.graphx.impl.GraphImpl
import org.apache.spark.storage.StorageLevel


private[ml] class VMBLPPartitioner(numParts: Int) extends Partitioner {

  override def numPartitions: Int = numParts

  def getPartition(key: Any): PartitionID = {
    key.asInstanceOf[Int]
  }

  override def equals(other: Any): Boolean = other match {
    case h: VMBLPPartitioner =>
      h.numPartitions == numPartitions
    case _ =>
      false
  }

  override def hashCode: Int = numPartitions
}

object VMBLPPartitioner {
  type PVD = AliasTable[Int]

  /**
   * Stochastic Balanced Label Propogation, see:
   * https://code.facebook.com/posts/274771932683700/large-scale-graph-partitioning-with-apache-giraph/
   * This is the vertex-cut version (SBLP is an edge-cut algorithm for Apache Giraph)
   */
  private[zen] def partitionByVMBLP[VD: ClassTag, ED: ClassTag](
    input: Graph[VD, ED],
    numIter: Int,
    storageLevel: StorageLevel): Graph[VD, ED] = {
    val numPartitions = input.edges.partitions.length
    val vmblp = new VMBLPPartitioner(numPartitions)
    val gen = new XORShiftRandom()

    var pidGraph = input.mapEdges((pid, iter) => iter.map(t => pid))
      .mapVertices[AliasTable[Int]]((_, _) => null)
    pidGraph.persist(storageLevel)

    for (iter <- 1 to numIter) {
      val transCounter = pidGraph.edges.mapPartitions(_.flatMap(edge => {
        val pid = edge.attr
        Iterator((edge.srcId, pid), (edge.dstId, pid))
      })).aggregateByKey(BSV.zeros[Int](numPartitions), pidGraph.vertices.partitioner.get)((agg, pid) => {
        agg(pid) += 1
        agg
      }, _ :+= _)
      pidGraph.joinVertices(transCounter)((_, _, counter) => AliasTable.generateAlias(counter))

      val transGraph = pidGraph.mapTriplets(triplet => {
        val table1 = triplet.srcAttr
        val table2 = triplet.dstAttr
        val rand = gen.nextInt(table1.norm + table2.norm)
        val toPid = if (rand < table1.norm) {
          table1.sampleFrom(rand, gen)
        } else {
          table2.sampleFrom(rand - table1.norm, gen)
        }
        (triplet.attr, toPid)
      })
      val newGraph = GraphImpl(transGraph.vertices.mapValues[AliasTable[Int]](t => null), transGraph.edges)
      val transMat = newGraph.edges.aggregate(DenseMatrix.zeros[Long](numPartitions, numPartitions))((agg, edge) => {
          agg(edge.attr) += 1
          agg
        }, _ :+= _)
      val rateMat = DenseMatrix.zeros[Float](numPartitions, numPartitions)
      for (i <- 0 until numPartitions) {
        for (j <- i+1 until numPartitions) {
          val numOut = transMat(i, j)
          val numIn = transMat(j, i)
          val thershold = math.min(numOut, numIn)
          val numDelta = transMat(i, i) + numOut - (transMat(j, j) + numIn)
          rateMat(i, j) = ((numDelta / 2 + thershold) / numOut.toDouble).toFloat
          rateMat(j, i) = ((-numDelta / 2) + thershold / numIn.toDouble).toFloat
        }
      }
      pidGraph = newGraph.mapEdges(edge => {
        val (pid, toPid) = edge.attr
        if (gen.nextFloat() < rateMat(pid, toPid)) toPid else pid
      })
    }

    val a = input.edges.innerJoin(pidGraph.edges)((_, _, pid, edge) => (pid, edge))
      .mapPartitions(iter => iter.map(e => (e, Edge(e.srcId, e.dstId, e.attr)))).partitionBy(vmblp).map(_._2)
    GraphImpl(input.vertices, a, null.asInstanceOf[VD], storageLevel, storageLevel)
  }
}
