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

import java.util.concurrent.CountDownLatch

import LDA._
import LDADefines._
import breeze.linalg.sum
import org.apache.log4j.Logger
import org.apache.spark.graphx2.impl.GraphImpl

object LDAMetrics {
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
  def perplexity(lda: LDA): Double = {
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

    val newSvps = vertices.partitionsRDD.mapPartitions(_.map(svp => {
      val totalSize = svp.capacity
      val newValues = new Array[(TC, Double, Int)](totalSize)
      val sizePerThrd = {
        val npt = totalSize / numThreads
        if (npt * numThreads == totalSize) npt else npt + 1
      }
      val doneSignal = new CountDownLatch(numThreads)
      val threads = new Array[Thread](numThreads)
      for (threadId <- threads.indices) {
        threads(threadId) = new Thread(new Runnable {
          val startPos = sizePerThrd * threadId
          val endPos = math.min(sizePerThrd * (threadId + 1), totalSize)

          override def run(): Unit = {
            val logger = Logger.getLogger(this.getClass.getName)
            val mask = svp.mask
            val index = svp.index
            val values = svp.values
            try {
              var i = mask.nextSetBit(startPos)
              while (i < endPos && i >= 0) {
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
                newValues(i) = (counter, pSum, cSum)
                i = mask.nextSetBit(i + 1)
              }
            } catch {
              case e: Exception => logger.error(e.getLocalizedMessage, e)
            } finally {
              doneSignal.countDown()
            }
          }
        }, s"preComputing thread $threadId")
      }
      threads.foreach(_.start())
      doneSignal.await()
      svp.withValues(newValues)
    }), preservesPartitioning=true)
    val newVerts = vertices.withPartitionsRDD(newSvps)
    val cachedGraph = refreshEdgeAssociations(graph, newVerts)
    val pplxEdges = cachedGraph.edges.partitionsRDD.mapPartitions(_.map(t => {
      val ep = t._2
      val totalSize = ep.size
      val sizePerThrd = {
        val npt = totalSize / numThreads
        if (npt * numThreads == totalSize) npt else npt + 1
      }
      val sums = new Array[Double](numThreads)
      val doneSignal = new CountDownLatch(numThreads)
      val threads = new Array[Thread](numThreads)
      for (threadId <- threads.indices) {
        threads(threadId) = new Thread(new Runnable {
          val thid = threadId
          val startPos = sizePerThrd * threadId
          val endPos = math.min(sizePerThrd * (threadId + 1), totalSize)

          override def run(): Unit = {
            val logger = Logger.getLogger(this.getClass.getName)
            val lcSrcIds = ep.localSrcIds
            val lcDstIds = ep.localDstIds
            val vattrs = ep.vertexAttrs
            val data = ep.data
            try {
              for (i <- startPos until endPos) {
                val (termTopicCounter, wSparseSum, _) = vattrs(lcSrcIds(i))
                val (docTopicCounter, dSparseSum, docSize) = vattrs(lcDstIds(i))
                val occurs = data(i).length

                // \frac{{n}_{kw}{n}_{kd}}{{n}_{k}+\bar{\beta}}
                val dwSparseSum = docTopicCounter.activeIterator.map {
                  case (topic, cnt) => cnt * termTopicCounter(topic) / (tCounter(topic) + betaSum)
                }.sum
                val prob = (tDenseSum + wSparseSum + dSparseSum + dwSparseSum) / (docSize + alphaSum)

                sums(thid) += Math.log(prob) * occurs
              }
            } catch {
              case e: Exception => logger.error(e.getLocalizedMessage, e)
            } finally {
              doneSignal.countDown()
            }
          }
        }, s"perplexity thread $threadId")
      }
      threads.foreach(_.start())
      doneSignal.await()
      sums.sum
    }), preservesPartitioning=true)

    val termProb = pplxEdges.sum()
    math.exp(-1 * termProb / numTokens)
  }
}
