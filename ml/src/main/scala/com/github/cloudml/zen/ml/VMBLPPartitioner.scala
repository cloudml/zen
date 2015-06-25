package com.github.cloudml.zen.ml

import scala.reflect.ClassTag
import org.apache.spark.HashPartitioner
import org.apache.spark.graphx._
import org.apache.spark.graphx.impl.GraphImpl
import org.apache.spark.storage.StorageLevel

object VMBLPPartitioner {
  /**
   * Modified Balanced Label Propogation, see: https://code.facebook.com/posts/274771932683700/large-scale-graph-partitioning-with-apache-giraph/
   * This is the vertex-cut version (MBLP is an edge-cut algorithm for Apache Giraph)
   */
  def partitionByVMBLP[VD: ClassTag, ED: ClassTag](
      inGraph: Graph[VD, ED],
      numIter: Int,
      storageLevel: StorageLevel): Graph[VD, ED] = {
    
    val numPartitions = inGraph.edges.partitions.length
    var tbrGraph = inGraph
    tbrGraph.persist(storageLevel)
    
    for (i <- 0 to numIter) {
      val pidRdd = tbrGraph.vertices.mapPartitionsWithIndex((pid, iter) => iter.map(t => (t._1, pid)), true)
      val pidVertices = VertexRDD(pidRdd)  // Get Vertices which v.attr = <partitionId of v>
      
      val pidGraph = GraphImpl(pidVertices, tbrGraph.edges)
      val neiVecVertices = pidGraph.aggregateMessages[Array[Int]](ectx => {
        val vecSrc = Array.fill(numPartitions)(0)
        vecSrc(ectx.dstAttr) += 1
        val vecDst = Array.fill(numPartitions)(0)
        vecDst(ectx.srcAttr) += 1
        ectx.sendToSrc(vecSrc)
        ectx.sendToDst(vecDst)
      }, (_, _).zipped.map(_ + _))  // Get Vertices which v.attr = Array[d0, d1, ..., dn]
      
      val wantVertices = neiVecVertices.mapValues(discreteSample(_))
      // Get Vertices which v.attr = (<partitionId now>, <partitionId to move to>) 
      val moveVertices = pidVertices.innerZipJoin(wantVertices)((_, fromPid, toPid) => (fromPid, toPid))
      
      // Get a matrix which mat(i)(j) = total num of vertices that want to move from i to j
      val moveMat = moveVertices.aggregate(Array.fill(numPartitions, numPartitions)(0))({
        case (mat, (_, ft)) => {
          if(ft._1 != ft._2) mat(ft._1)(ft._2) += 1
          mat
      }}, (_, _).zipped.map((_, _).zipped.map(_ + _)))
      
      val newPidRdd = moveVertices.mapPartitions(iter => iter.map({case (vid, (from, to)) => {
        if(from == to) (vid, from)
        else {
          // Move vertices under balance constraints
          val numOut = moveMat(from)(to)
          val numIn = moveMat(to)(from)
          val threshold = math.min(numOut, numIn)
          val r = threshold.asInstanceOf[Float] / numOut
          val u = math.random
          if(u < r) (vid, to)
          else (vid, from)
        }
      }}))
      val newPidVertices = VertexRDD(newPidRdd)  // Get Vertices which v.attr = <partitionId after moving>
      
      // Repartition
      val newPidGraph = GraphImpl(newPidVertices, tbrGraph.edges)
      val tempEdges = newPidGraph.triplets.mapPartitions(iter => iter.map{
        et => (et.srcAttr, Edge(et.srcId, et.dstId, et.attr))
      })
      val newEdges = tempEdges.partitionBy(new HashPartitioner(numPartitions)).map(_._2)
      
      val ntbrGraph = GraphImpl(tbrGraph.vertices, newEdges, null.asInstanceOf[VD], storageLevel, storageLevel)
      ntbrGraph.persist(storageLevel)
      tbrGraph.unpersist(false)
      tbrGraph = ntbrGraph
    }  // End for
    tbrGraph
  }
  
  def discreteSample(dist: Array[Int]): Int = {
    val s = dist.sum
    val u = math.random * s
    var ps = 0
    for(p <- dist) {
      ps += p
      if(u < ps) return p
    }
    dist.length - 1
  }
}
