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

import java.util.concurrent.{ConcurrentLinkedQueue, Executors}

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV}
import breeze.numerics.lgamma
import com.github.cloudml.zen.ml.clustering.LDADefines._
import com.github.cloudml.zen.ml.clustering.{LDALogLikelihood, LDAPerplexity}
import com.github.cloudml.zen.ml.util.BVDecompressor
import org.apache.spark.graphx2.impl.{ShippableVertexPartition => VertPartition, _}

import scala.collection.JavaConversions._
import scala.concurrent._
import scala.concurrent.duration._
import scala.language.existentials


abstract class LDAAlgorithm(numTopics: Int,
  numThreads: Int) extends Serializable {
  protected val dscp = numTopics >>> 3

  def isByDoc: Boolean

  def samplePartition(numPartitions: Int,
    sampIter: Int,
    seed: Int,
    topicCounters: BDV[Count],
    numTokens: Long,
    numTerms: Int,
    alpha: Double,
    alphaAS: Double,
    beta: Double)
    (pid: Int, ep: EdgePartition[TA, Nvk]): EdgePartition[TA, Int]

  def countPartition(ep: EdgePartition[TA, Int]): Iterator[NvkPair]

  def aggregateCounters(vp: VertPartition[TC], cntsIter: Iterator[NvkPair]): VertPartition[TC]

  def perplexPartition(topicCounters: BDV[Count],
    numTokens: Long,
    numTerms: Int,
    alpha: Double,
    alphaAS: Double,
    beta: Double)
    (ep: EdgePartition[TA, Nvk]): (Double, Double, Double)

  def logLikelihoodPartition(topicCounters: BDV[Count],
    numTokens: Long,
    alpha: Double,
    beta: Double,
    alphaAS: Double)
    (vp: VertPartition[TC]): (Double, Double)

  def initEdgePartition(ep: EdgePartition[TA, _]): EdgePartition[TA, Int] = {
    ep.withVertexAttributes(new Array[Int](ep.vertexAttrs.length))
  }

  def sampleGraph(edges: EdgeRDDImpl[TA, _],
    verts: VertexRDDImpl[TC],
    topicCounters: BDV[Count],
    seed: Int,
    sampIter: Int,
    numTokens: Long,
    numTerms: Int,
    alpha: Double,
    alphaAS: Double,
    beta: Double): EdgeRDDImpl[TA, Int] = {
    val newEdges = refreshEdgeAssociations(edges, verts)
    val numPartitions = newEdges.partitions.length
    val spf = samplePartition(numPartitions, sampIter, seed, topicCounters,
      numTokens, numTerms, alpha, alphaAS, beta)_
    val partRDD = newEdges.partitionsRDD.mapPartitions(_.map { case (pid, ep) =>
      val startedAt = System.nanoTime
      val newEp = spf(pid, ep)
      val elapsedSeconds = (System.nanoTime - startedAt) / 1e9
      println(s"Partition sampling $sampIter takes: $elapsedSeconds secs")
      (pid, newEp)
    }, preservesPartitioning=true)
    newEdges.withPartitionsRDD(partRDD)
  }

  def updateVertexCounters(edges: EdgeRDDImpl[TA, Int],
    verts: VertexRDDImpl[TC]): VertexRDDImpl[TC] = {
    val shippedCounters = edges.partitionsRDD.mapPartitions(_.flatMap { case (_, ep) =>
      countPartition(ep)
    }).partitionBy(verts.partitioner.get)

    // Below identical map is used to isolate the impact of locality of CheckpointRDD
    val isoRDD = verts.partitionsRDD.mapPartitions(_.seq, preservesPartitioning=true)
    val partRDD = isoRDD.zipPartitions(shippedCounters, preservesPartitioning=true)(
      (vpIter, cntsIter) => vpIter.map(aggregateCounters(_, cntsIter))
    )
    verts.withPartitionsRDD(partRDD)
  }

  def calcPerplexity(edges: EdgeRDDImpl[TA, _],
    verts: VertexRDDImpl[TC],
    topicCounters: BDV[Count],
    numTokens: Long,
    numTerms: Int,
    alpha: Double,
    alphaAS: Double,
    beta: Double): LDAPerplexity = {
    val newEdges = refreshEdgeAssociations(edges, verts)
    val ppf = perplexPartition(topicCounters, numTokens, numTerms, alpha, alphaAS, beta)_
    val sumPart = newEdges.partitionsRDD.mapPartitions(_.map { case (_, ep) =>
      ppf(ep)
    })
    val (llht, wllht, dllht) = sumPart.collect().unzip3
    val pplx = math.exp(-llht.par.sum / numTokens)
    val wpplx = math.exp(-wllht.par.sum / numTokens)
    val dpplx = math.exp(-dllht.par.sum / numTokens)
    new LDAPerplexity(pplx, wpplx, dpplx)
  }

  def calcLogLikelihood(verts: VertexRDDImpl[TC],
    topicCounters: BDV[Count],
    numTokens: Long,
    numDocs: Long,
    numTerms: Int,
    alpha: Double,
    alphaAS: Double,
    beta: Double): LDALogLikelihood = {
    val alphaSum = alpha * numTopics
    val betaSum = beta * numTerms
    val lpf = logLikelihoodPartition(topicCounters, numTokens, alpha, beta, alphaAS)_
    val sumPart = verts.partitionsRDD.mapPartitions(_.map(lpf))
    val (wllht, dllht) = sumPart.collect().unzip
    val normWord = Range(0, numTopics).par.map(i => lgamma(topicCounters(i) + betaSum)).sum
    val wllh = wllht.par.sum + numTopics * lgamma(betaSum) - normWord
    val dllh = dllht.par.sum + numDocs * lgamma(alphaSum)
    new LDALogLikelihood(wllh, dllh)
  }

  def refreshEdgeAssociations(edges: EdgeRDDImpl[TA, _],
    verts: VertexRDDImpl[TC]): EdgeRDDImpl[TA, Nvk] = {
    val shippedVerts = verts.partitionsRDD.mapPartitions(_.flatMap(vp => {
      val rt = vp.routingTable
      val index = vp.index
      val values = vp.values
      implicit val es = initPartExecutionContext()
      Range(0, rt.numEdgePartitions).grouped(numThreads).flatMap(batch => {
        val all = Future.traverse(batch) { pid =>
          val future = Future {
            val vids = rt.routingTable(pid)._1
            val attrs = vids.map(vid => values(index.getPos(vid)))
            (pid, new VertexAttributeBlock(vids, attrs))
          }
          future.onFailure { case e =>
            e.printStackTrace()
          }
          future
        }
        Await.result(all, 1.hour)
      }) ++ {
        closePartExecutionContext()
        Iterator.empty
      }
    })).partitionBy(edges.partitioner.get)

    // Below identical map is used to isolate the impact of locality of CheckpointRDD
    val isoRDD = edges.partitionsRDD.mapPartitions(_.seq, preservesPartitioning=true)
    val partRDD = isoRDD.zipPartitions(shippedVerts, preservesPartitioning=true)(
      (epIter, vabsIter) => epIter.map(Function.tupled((pid, ep) => {
        val g2l = ep.global2local
        val results = new Array[Nvk](ep.vertexAttrs.length)
        val thq = new ConcurrentLinkedQueue(0 until numThreads)
        val decomps = Array.fill(numThreads)(new BVDecompressor(numTopics))

        implicit val es = initPartExecutionContext()
        val all = Future.traverse(vabsIter) { case (_, vab) =>
          val future = Future {
            val thid = thq.poll()
            val decomp = decomps(thid)
            vab.iterator.foreach { case (vid, vdata) =>
              results(g2l(vid)) = decomp.CV2BV(vdata)
            }
            thq.add(thid)
          }
          future.onFailure { case e =>
            e.printStackTrace()
          }
          future
        }
        Await.ready(all, 1.hour)
        closePartExecutionContext()

        (pid, ep.withVertexAttributes(results))
      }))
    )
    edges.withPartitionsRDD(partRDD)
  }

  def collectTopicCounters(verts: VertexRDDImpl[TC]): BDV[Count] = {
    verts.partitionsRDD.mapPartitions(_.map(vp => {
      val totalSize = vp.capacity
      val index = vp.index
      val mask = vp.mask
      val values = vp.values
      val sizePerthrd = {
        val npt = totalSize / numThreads
        if (npt * numThreads == totalSize) npt else npt + 1
      }

      implicit val es = initPartExecutionContext()
      val all = Future.traverse(Range(0, numThreads).iterator) { thid =>
        val future = Future {
          val decomp = new BVDecompressor(numTopics)
          val startPos = sizePerthrd * thid
          val endPos = math.min(sizePerthrd * (thid + 1), totalSize)
          val agg = new BDV(new Array[Count](numTopics))
          var pos = mask.nextSetBit(startPos)
          while (pos < endPos && pos >= 0) {
            if (isTermId(index.getValue(pos))) {
              val bv = decomp.CV2BV(values(pos))
              bv match {
                case v: BDV[Count] => agg :+= v
                case v: BSV[Count] => agg :+= v
              }
            }
            pos = mask.nextSetBit(pos + 1)
          }
          agg
        }
        future.onFailure { case e =>
          e.printStackTrace()
        }
        future
      }
      val aggs = Await.result(all, 1.hour)
      closePartExecutionContext()

      aggs.reduce(_ :+= _)
    })).collect().par.reduce(_ :+= _)
  }

  // note: should only be called in one partition!
  protected def initPartExecutionContext(): ExecutionContextExecutorService = {
    ExecutionContext.fromExecutorService(Executors.newFixedThreadPool(numThreads))
  }

  // note: should be called after finish using ExecutionContext
  protected def closePartExecutionContext()(implicit es: ExecutionContextExecutorService): Unit = {
    es.shutdown()
  }
}
