package com.github.cloudml.zen.ml

import org.apache.spark.Partitioner
import org.apache.spark.graphx._

import scala.math._

/**
 * Degree-Based Hashing, the paper:
 * Distributed Power-law Graph Computing: Theoretical and Empirical Analysis
 */
private[ml] class DBHPartitioner[ED](val partitions: Int, val threshold: Int = 70) extends Partitioner {
  val mixingPrime: Long = 1125899906842597L

  def numPartitions = partitions

  /*
   * default Degree Based Hashing,
     "Distributed Power-law Graph Computing: Theoretical and Empirical Analysis"
    def getPartition(key: Any): Int = {
      val edge = key.asInstanceOf[EdgeTriplet[Int, ED]]
      val srcDeg = edge.srcAttr
      val dstDeg = edge.dstAttr
      val srcId = edge.srcId
      val dstId = edge.dstId
      if (srcDeg < dstDeg) {
        getPartition(srcId)
      } else {
        getPartition(dstId)
      }
    }
   */

  /**
   * Default DBH doesn't consider the situation where both the degree of src and
   * dst vertices are both small than a given threshold value
   */
  def getPartition(key: Any): Int = {
    val edge = key.asInstanceOf[EdgeTriplet[Int, ED]]
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
    case h: DBHPartitioner[ED] =>
      h.numPartitions == numPartitions
    case _ =>
      false
  }

  override def hashCode: Int = numPartitions
}

