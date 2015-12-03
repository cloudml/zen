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
import breeze.linalg.{DenseMatrix, SparseVector => BSV}
import com.github.cloudml.zen.ml.sampler.FTree
import com.github.cloudml.zen.ml.util.XORShiftRandom
import org.apache.spark.Partitioner
import org.apache.spark.graphx2._
import org.apache.spark.graphx2.impl.GraphImpl
import org.apache.spark.storage.StorageLevel


private[ml] class VSDLPPartitioner(numParts: Int) extends Partitioner {

  override def numPartitions: Int = numParts

  def getPartition(key: Any): PartitionID = {
    key.asInstanceOf[PartitionID] % numPartitions
  }

  override def equals(other: Any): Boolean = other match {
    case vsdlp: VSDLPPartitioner =>
      vsdlp.numPartitions == numPartitions
    case _ =>
      false
  }

  override def hashCode: Int = numPartitions
}

/**
 * Stochastic Balanced Label Propogation, see:
 * https://code.facebook.com/posts/274771932683700/large-scale-graph-partitioning-with-apache-giraph/
 * This is the vertex-cut version (SBLP is an edge-cut algorithm for Apache Giraph), with dynamic transferring
 */
object VSDLPPartitioner {
  type PVD = FTree[Int]

  private[zen] def partitionByVSDLP[VD: ClassTag, ED: ClassTag](
    input: Graph[VD, ED],
    numIter: Int,
    storageLevel: StorageLevel): Graph[VD, ED] = {
    val edges = input.edges
    val conf = edges.context.getConf
    val numPartitions = conf.getInt(cs_numPartitions, edges.partitions.length)
    val vsdlp = new VSDLPPartitioner(numPartitions)

    var pidGraph = input.mapEdges((pid, iter) => iter.map(_ => pid)).mapVertices[PVD]((_, _) => null)
    pidGraph.persist(storageLevel)
    for (iter <- 1 to numIter) {
      val transCounter = pidGraph.edges.mapPartitions(_.flatMap(edge => {
        val pid = edge.attr
        Iterator((edge.srcId, pid), (edge.dstId, pid))
      })).aggregateByKey(BSV.zeros[Int](numPartitions), pidGraph.vertices.partitioner.get)((agg, pid) => {
        agg(pid) += 1
        agg
      }, _ :+= _)

      val transGraph = pidGraph.joinVertices(transCounter)((_, _, counter) => FTree.generateFTree(counter))
        .mapTriplets((pid, iter) => {
        val gen = new XORShiftRandom()
        iter.map(et => {
          val ftree1 = et.srcAttr
          val ftree2 = et.dstAttr
          val pid = et.attr
          ftree1.deltaUpdate(pid, -1)
          ftree2.deltaUpdate(pid, -1)
          val norm1 = ftree1.norm
          val norm2 = ftree2.norm
          val toPid = if (norm1 == 0 && norm2 == 0) {
            gen.nextInt(numPartitions)
          } else {
            val u = gen.nextInt(norm1 + norm2)
            if (u < norm1) {
              ftree1.sampleFrom(u, gen)
            } else {
              ftree2.sampleFrom(u - norm1, gen)
            }
          }
          ftree1.deltaUpdate(pid, 1)
          ftree2.deltaUpdate(pid, 1)
          (pid, toPid)
        })
      }, TripletFields.All).mapVertices[PVD]((_, _) => null)
      transGraph.persist(storageLevel)

      val transMat = transGraph.edges.aggregate(DenseMatrix.zeros[Long](numPartitions, numPartitions))((agg, edge) => {
        agg(edge.attr) += 1
        agg
      }, _ :+= _)
      val rateMat = DenseMatrix.zeros[Float](numPartitions, numPartitions)
      for (i <- 0 until numPartitions) {
        for (j <- i + 1 until numPartitions) {
          val numOut = transMat(i, j)
          val numIn = transMat(j, i)
          val thershold = math.min(numOut, numIn)
          val numDelta = transMat(i, i) + numOut - (transMat(j, j) + numIn)
          rateMat(i, j) = ((numDelta / (iter + 1) + thershold) / numOut.toDouble).toFloat
          rateMat(j, i) = ((-numDelta / (iter + 1) + thershold) / numIn.toDouble).toFloat
        }
      }
      pidGraph = transGraph.mapEdges((pid, iter) => {
        val gen = new XORShiftRandom()
        iter.map(edge => {
          val (pid, toPid) = edge.attr
          if (gen.nextFloat() < rateMat(pid, toPid)) toPid else pid
        })
      })
      pidGraph.persist(storageLevel)
    }

    val newEdges = edges.innerJoin(pidGraph.edges)((_, _, ed, toPid) => (toPid, ed))
      .mapPartitions(_.map(e =>(e.attr._1, Edge(e.srcId, e.dstId, e.attr._2))))
      .partitionBy(vsdlp).map(_._2)
    GraphImpl(input.vertices, newEdges, null.asInstanceOf[VD], storageLevel, storageLevel)
  }
}
