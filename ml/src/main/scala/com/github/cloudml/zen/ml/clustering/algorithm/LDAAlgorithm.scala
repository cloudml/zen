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

package com.github.cloudml.zen.ml.clustering.algorithm

import breeze.linalg.{DenseVector => BDV}
import com.github.cloudml.zen.ml.clustering.LDADefines._
import com.github.cloudml.zen.ml.clustering.LDAPerplexity
import org.apache.spark.graphx2.impl.{ShippableVertexPartition => VertPartition, EdgePartition, GraphImpl}


abstract class LDAAlgorithm(numTopics: Int, numThreads: Int)
  extends Serializable {
  protected val dscp = numTopics >> 3

  def isByDoc: Boolean

  def samplePartition(accelMethod: String,
    numPartitions: Int,
    sampIter: Int,
    seed: Int,
    topicCounters: BDV[Count],
    numTokens: Long,
    numTerms: Int,
    alpha: Double,
    alphaAS: Double,
    beta: Double)
    (pid: Int, ep: EdgePartition[TA, TC]): EdgePartition[TA, TC]

  def countPartition(ep: EdgePartition[TA, TC]): Iterator[NvkPair]

  def aggregateCounters(vp: VertPartition[TC], cntsIter: Iterator[NvkPair]): VertPartition[TC]

  def perplexPartition(topicCounters: BDV[Count],
    numTokens: Long,
    numTerms: Int,
    alpha: Double,
    alphaAS: Double,
    beta: Double)
    (ep: EdgePartition[TA, TC]): (Double, Double, Double)

  def sampleGraph(corpus: GraphImpl[TC, TA],
    topicCounters: BDV[Count],
    seed: Int,
    sampIter: Int,
    numTokens: Long,
    numTerms: Int,
    alpha: Double,
    alphaAS: Double,
    beta: Double): GraphImpl[TC, TA] = {
    val graph = refreshEdgeAssociations(corpus)
    val edges = graph.edges
    val vertices = graph.vertices
    val conf = edges.context.getConf
    val accelMethod = conf.get(cs_accelMethod, "alias")
    val numPartitions = edges.partitions.length
    val spf = samplePartition(accelMethod, numPartitions, sampIter, seed,
      topicCounters, numTokens, numTerms, alpha, alphaAS, beta)_
    val partRDD = edges.partitionsRDD.mapPartitions(_.map(Function.tupled((pid, ep) => {
      val startedAt = System.nanoTime()
      val newEp = spf(pid, ep)
      val elapsedSeconds = (System.nanoTime() - startedAt) / 1e9
      println(s"Partition sampling $sampIter takes: $elapsedSeconds secs")
      (pid, newEp)
    })), preservesPartitioning=true)
    GraphImpl.fromExistingRDDs(vertices, edges.withPartitionsRDD(partRDD))
  }

  def updateVertexCounters(sampledCorpus: GraphImpl[TC, TA]): GraphImpl[TC, TA] = {
    val vertices = sampledCorpus.vertices
    val edges = sampledCorpus.edges
    val shippedCounters = edges.partitionsRDD.mapPartitions(_.flatMap(Function.tupled((_, ep) =>
      countPartition(ep)
    ))).partitionBy(vertices.partitioner.get)

    // Below identical map is used to isolate the impact of locality of CheckpointRDD
    val isoRDD = vertices.partitionsRDD.mapPartitions(iter => iter, preservesPartitioning=true)
    val partRDD = isoRDD.zipPartitions(shippedCounters, preservesPartitioning=true)(
      (vpIter, cntsIter) => vpIter.map(vp => aggregateCounters(vp, cntsIter))
    )
    GraphImpl.fromExistingRDDs(vertices.withPartitionsRDD(partRDD), edges)
  }

  def calcPerplexity(corpus: GraphImpl[TC, TA],
    topicCounters: BDV[Count],
    numTokens: Long,
    numTerms: Int,
    alpha: Double,
    alphaAS: Double,
    beta: Double): LDAPerplexity = {
    val refrGraph = refreshEdgeAssociations(corpus)
    val edges = refrGraph.edges
    val ppf = perplexPartition(topicCounters, numTokens, numTerms, alpha, alphaAS, beta)_
    val sumPart = edges.partitionsRDD.mapPartitions(_.map(Function.tupled((_, ep) =>
      ppf(ep)
    )))
    val (llht, wllht, dllht) = sumPart.collect().unzip3
    val pplx = math.exp(-llht.par.sum / numTokens)
    val wpplx = math.exp(-wllht.par.sum / numTokens)
    val dpplx = math.exp(-dllht.par.sum / numTokens)
    new LDAPerplexity(pplx, wpplx, dpplx)
  }
}
