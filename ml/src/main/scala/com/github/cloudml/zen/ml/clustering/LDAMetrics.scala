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
import scala.concurrent.duration._

import LDADefines._

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV}
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
      val totalSize = svp.capacity
      val index = svp.index
      val mask = svp.mask
      val values = svp.values
      val results = new Array[(TC, Double, Int)](totalSize)
      val sizePerthrd = {
        val npt = totalSize / numThreads
        if (npt * numThreads == totalSize) npt else npt + 1
      }
      implicit val es = ExecutionContext.fromExecutorService(Executors.newFixedThreadPool(numThreads))
      val all = Range(0, numThreads).map(thid => Future {
        val startPos = sizePerthrd * thid
        val endPos = math.min(sizePerthrd * (thid + 1), totalSize)
        var pos = mask.nextSetBit(startPos)
        while (pos < endPos && pos >= 0) {
          val vid = index.getValue(pos)
          val counter = values(pos)
          def itemProb(topic: Int, cnt: Count) = if (isDocId(vid)) {
            cnt * beta / (topicCounters(topic) + betaSum)
          } else {
            cnt * alphaRatio * (topicCounters(topic) + alphaAS) / (topicCounters(topic) + betaSum)
          }
          val pSum = counter.activeIterator.filter(_._2 > 0).map(Function.tupled(itemProb)).sum
          val cSum = if (isDocId(vid)) counter.activeValuesIterator.sum else 0
          results(pos) = (counter, pSum, cSum)
          pos = mask.nextSetBit(pos + 1)
        }
      })
      Await.ready(Future.sequence(all), 1.hour)
      es.shutdown()
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
      @volatile var llhs = 0D
      @volatile var wllhs = 0D
      @volatile var dllhs = 0D

      implicit val es = ExecutionContext.fromExecutorService(Executors.newFixedThreadPool(numThreads))
      val all = Future.traverse(ep.index.iterator)(t => Future {
        var pos = t._2
        val si = lcSrcIds(pos)
        val (orgTermTopics, wSparseSum, _) = vattrs(si).asInstanceOf[(TC, Double, Int)]
        val termTopics = orgTermTopics match {
          case v: BDV[Count] => v
          case v: BSV[Count] => toBDV(v)
        }
        var llhs_th = 0D
        var wllhs_th = 0D
        var dllhs_th = 0D
        while (pos < totalSize && lcSrcIds(pos) == si) {
          val (docTopics, dSparseSum, docSize) = vattrs(lcDstIds(pos)).asInstanceOf[(BSV[Count], Double, Int)]
          val topics = data(pos)
          // \frac{{n}_{kw}{n}_{kd}}{{n}_{k}+\bar{\beta}}
          val dwSparseSum = docTopics.activeIterator.map(Function.tupled((topic, cnt) =>
            cnt * termTopics(topic) / (topicCounters(topic) + betaSum)
          )).sum
          val prob = (tDenseSum + wSparseSum + dSparseSum + dwSparseSum) / (docSize + alphaSum)
          llhs_th += Math.log(prob) * topics.length
          for (topic <- topics) {
            val wProb = (termTopics(topic) + beta) / (topicCounters(topic) + betaSum)
            val dProb = (docTopics(topic) + alphaRatio * (topicCounters(topic) + alphaAS)) /
              (docSize + alphaSum)
            wllhs_th += Math.log(wProb)
            dllhs_th += Math.log(dProb)
          }
          pos += 1
        }
        llhs += llhs_th
        wllhs += wllhs_th
        dllhs += dllhs_th
      })
      Await.ready(all, Duration.Inf)
      es.shutdown()

      (llhs, wllhs, dllhs)
    }))

    val (llht, wllht, dllht) = sumPart.collect().unzip3
    pplx = math.exp(-llht.par.sum / numTokens)
    wpplx = math.exp(-wllht.par.sum / numTokens)
    dpplx = math.exp(-dllht.par.sum / numTokens)
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
