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
import scala.concurrent._
import scala.concurrent.duration.Duration

import LDADefines._

import breeze.linalg.sum
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
    val tCounter = lda.totalTopicCounter
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
    val tDenseSum = tCounter.valuesIterator.map(cnt =>
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
              cnt * beta / (tCounter(topic) + betaSum)
            } else {
              cnt * alphaRatio * (tCounter(topic) + alphaAS) / (tCounter(topic) + betaSum)
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
      implicit val ec = ExecutionContext.fromExecutorService(Executors.newFixedThreadPool(numThreads))
      val all = Future.traverse(ep.index.iterator)(t => Future {
        var pos = t._2
        val lcVid = lcSrcIds(pos)
        val (termTopicCounter, wSparseSum, _) = vattrs(lcVid)
        var lcSum = 0D
        while (pos < totalSize && lcSrcIds(pos) == lcVid) {
          val (docTopicCounter, dSparseSum, docSize) = vattrs(lcDstIds(pos))
          val occurs = data(pos).length
          // \frac{{n}_{kw}{n}_{kd}}{{n}_{k}+\bar{\beta}}
          val dwSparseSum = docTopicCounter.activeIterator.map {
            case (topic, cnt) => cnt * termTopicCounter(topic) / (tCounter(topic) + betaSum)
          }.sum
          val prob = (tDenseSum + wSparseSum + dSparseSum + dwSparseSum) / (docSize + alphaSum)
          lcSum += Math.log(prob) * occurs
          pos += 1
        }
        lcSum
      })
      val sums = all.map(_.sum)
      val result = Await.result(sums, Duration.Inf)
      ec.shutdown()
      result
    }))

    val termProb = sumPart.sum()
    pplx = math.exp(-1 * termProb / numTokens)
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
