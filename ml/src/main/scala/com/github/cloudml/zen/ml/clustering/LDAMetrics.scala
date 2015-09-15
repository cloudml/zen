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

package com.github.cloudml.zen.ml.clustering

import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger
import scala.concurrent._
import scala.concurrent.duration.Duration

import LDADefines._

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, sum}
import org.apache.spark.graphx2.impl.GraphImpl


trait LDAMetrics {
  def measure(): Unit
}

class LDAPerplexity(lda: LDA) extends LDAMetrics {
  var pplx = 0D
  var wpplx = 0D
  var dpplx = 0D

  /**
   * the multiplcation between word distribution among all topics and the corresponding doc
   * distribution among all topics:
   * p(w)=\sum_{k}{p(k|d)*p(w|k)}=
   * \sum_{k}{\frac{{n}_{kw}+{\beta }_{w}} {{n}_{k}+\bar{\beta }} \frac{{n}_{kd}+{\alpha }_{k}}{\sum{{n}_{k}}+
   * \bar{\alpha }}}=
   * \sum_{k} \frac{{\alpha }_{k}{\beta }_{w}  + {n}_{kw}{\alpha }_{k} + {n}_{kd}{\beta }_{w} + {n}_{kw}{n}_{kd}}
   * {{n}_{k}+\bar{\beta }} \frac{1}{\sum{{n}_{k}}+\bar{\alpha }}}
   * \exp^{-(\sum{\log(p(w))})/N}
   * N is the number of tokens in corpus
   */
  def measure(): Unit = {
    val topicCounters = lda.topicCounters
    val numTopics = lda.numTopics
    val numTokens = lda.numTokens
    val alphaAS = lda.alphaAS
    val alphaSum = numTopics * lda.alpha
    val alphaRatio = alphaSum / (numTokens + alphaAS * numTopics)
    val beta = lda.beta
    val betaSum = lda.numTerms * beta

    val graph = lda.corpus.asInstanceOf[GraphImpl[TC, TA]]
    val vertices = graph.vertices
    val edges = graph.edges
    val numThreads = edges.context.getConf.getInt(cs_numThreads, 1)

    // \frac{{\alpha }_{k}{\beta }_{w}}{{n}_{k}+\bar{\beta }}
    val tDenseSum = topicCounters.valuesIterator.map(cnt =>
      beta * alphaRatio * (cnt + alphaAS) / (cnt + betaSum)
    ).sum

    val partRDD = vertices.partitionsRDD.mapPartitions(_.map(svp => {
      val index = svp.index
      val values= svp.values
      val results = new Array[(TC, Double, Int)](svp.capacity)
      implicit val ec = ExecutionContext.fromExecutorService(Executors.newFixedThreadPool(numThreads))
      val all = Future.traverse(svp.mask.iterator)(i => Future {
        val vid = index.getValue(i)
        val counter = values(i)
        val pSum = counter.activeIterator.map {
          case (topic, cnt) =>
            if (isDocId(vid)) {
              cnt * beta / (topicCounters(topic) + betaSum)
            } else {
              cnt * alphaRatio * (topicCounters(topic) + alphaAS) / (topicCounters(topic) + betaSum)
            }
        }.sum
        val cSum = if (isDocId(vid)) sum(counter) else 0
        results(i) = (counter, pSum, cSum)
      })
      Await.ready(all, Duration.Inf)
      ec.shutdown()
      svp.withValues(results)
    }), preservesPartitioning=true)
    val cachedGraph = GraphImpl.fromExistingRDDs(vertices.withPartitionsRDD(partRDD), edges)

    val refrGraph = refreshEdgeAssociations(cachedGraph)
    val sumPart = refrGraph.edges.partitionsRDD.mapPartitions(_.map(t => {
      val ep = t._2
      val totalSize = ep.size
      val lcSrcIds = ep.localSrcIds
      val lcDstIds = ep.localDstIds
      val vattrs = ep.vertexAttrs
      val data = ep.data
      val sums = new Array[Double](numThreads)
      val wSums = new Array[Double](numThreads)
      val dSums = new Array[Double](numThreads)
      val indicator = new AtomicInteger

      implicit val ec = ExecutionContext.fromExecutorService(Executors.newFixedThreadPool(numThreads))
      val all = Future.traverse(ep.index.iterator)(t => Future {
        val thid = indicator.getAndIncrement() % numThreads
        var pos = t._2
        val lcVid = lcSrcIds(pos)
        val (orgTermTopics, wSparseSum, _) = vattrs(lcVid)
        val termTopics = orgTermTopics match {
          case v: BDV[Count] => v
          case v: BSV[Count] => toBDV(v)
        }
        var lcSum = 0D
        var lcWSum = 0D
        var lcDSum = 0D
        while (pos < totalSize && lcSrcIds(pos) == lcVid) {
          val (docTopics, dSparseSum, docSize) = vattrs(lcDstIds(pos))
          val topics = data(pos)
          // \frac{{n}_{kw}{n}_{kd}}{{n}_{k}+\bar{\beta}}
          val dwSparseSum = docTopics.activeIterator.map {
            case (topic, cnt) => cnt * termTopics(topic) / (topicCounters(topic) + betaSum)
          }.sum
          val prob = (tDenseSum + wSparseSum + dSparseSum + dwSparseSum) / (docSize + alphaSum)
          lcSum += Math.log(prob) * topics.length
          for (topic <- topics) {
            val wProb = (termTopics(topic) + beta) / (topicCounters(topic) + betaSum)
            val dProb = (docTopics(topic) + alphaRatio * (topicCounters(topic) + alphaAS)) /
              (docSize + alphaSum)
            lcWSum += Math.log(wProb)
            lcDSum += Math.log(dProb)
          }
          pos += 1
        }
        sums(thid) += lcSum
        wSums(thid) += lcWSum
        dSums(thid) += lcDSum
      })
      Await.ready(all, Duration.Inf)
      ec.shutdown()
      (sums.sum, wSums.sum, dSums.sum)
    }))

    val (llh, wllh, dllh) = sumPart.collect().unzip3
    pplx = math.exp(-llh.sum / numTokens)
    wpplx = math.exp(-wllh.sum / numTokens)
    dpplx = math.exp(-dllh.sum / numTokens)
  }

  def getPerplexity: Double = pplx

  def getWordPerplexity: Double = wpplx

  def getDocPerplexity: Double = dpplx
}

object LDAPerplexity {
  def output(lda: LDA, writer: String => Unit): Unit = {
    val pplx = new LDAPerplexity(lda)
    pplx.measure()
    val o = s"perplexity=${pplx.getPerplexity}, word pplx=${pplx.getWordPerplexity}, doc pplx=${pplx.getDocPerplexity}"
    writer(o)
  }
}
