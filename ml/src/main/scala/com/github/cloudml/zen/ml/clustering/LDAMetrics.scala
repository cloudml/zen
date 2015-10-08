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
import java.util.concurrent.atomic.AtomicIntegerArray
import scala.concurrent._
import scala.concurrent.duration._

import LDADefines._
import LDAPerplexity._

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, convert, sum}


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
    val beta = lda.beta
    val betaSum = lda.numTerms * beta

    val refrGraph = refreshEdgeAssociations(lda.corpus)
    val edges = refrGraph.edges
    val numThreads = edges.context.getConf.getInt(cs_numThreads, 1)
    val sumPart = edges.partitionsRDD.mapPartitions(_.map(Function.tupled((_, ep) => {
      val alphaRatio = alphaSum / (numTokens + alphaAS * numTopics)
      val alphaK = (convert(topicCounters, Double) :+= alphaAS) :*= alphaRatio
      val denoms = BDV.tabulate(numTopics)(topic => 1D / (topicCounters(topic) + betaSum))
      val alphaK_denoms = (denoms.copy :*= ((alphaAS - betaSum) * alphaRatio)) :+= alphaRatio
      val beta_denoms = denoms.copy :*= beta
      // \frac{{\alpha }_{k}{\beta }_{w}}{{n}_{k}+\bar{\beta }}
      val tDenseSum = sum(alphaK_denoms.copy :*= beta)
      val totalSize = ep.size
      val lcSrcIds = ep.localSrcIds
      val lcDstIds = ep.localDstIds
      val vattrs = ep.vertexAttrs
      val data = ep.data
      val vertSize = vattrs.length
      val doc_denoms = new Array[Double](vertSize)
      val marks = new AtomicIntegerArray(vertSize)
      @volatile var llhs = 0D
      @volatile var wllhs = 0D
      @volatile var dllhs = 0D

      implicit val es = ExecutionContext.fromExecutorService(Executors.newFixedThreadPool(numThreads))
      val all = Future.traverse(ep.index.iterator)(Function.tupled((_, offset) => Future {
        val si = lcSrcIds(offset)
        val orgTermTopics = vattrs(si)
        val wSparseSum = calcWSparseSum(orgTermTopics, alphaK_denoms, numTopics)
        val twSum = tDenseSum + wSparseSum
        val termBeta_denoms = calcTermBetaDenoms(orgTermTopics, beta_denoms, denoms, numTopics)
        var llhs_th = 0D
        var wllhs_th = 0D
        var dllhs_th = 0D
        var pos = offset
        while (pos < totalSize && lcSrcIds(pos) == si) {
          val di = lcDstIds(pos)
          val docTopics = vattrs(di).asInstanceOf[BSV[Count]]
          if (marks.get(di) == 0) {
            doc_denoms(di) = 1D / (sum(docTopics) + alphaSum)
            marks.set(di, 1)
          }
          val doc_denom = doc_denoms(di)
          val topics = data(pos)
          val dwSum = calcDwSum(docTopics, termBeta_denoms)
          val prob = (twSum + dwSum) * doc_denom
          llhs_th += Math.log(prob) * topics.length
          for (topic <- topics) {
            wllhs_th += Math.log(termBeta_denoms(topic))
            dllhs_th += Math.log((docTopics(topic) + alphaK(topic)) * doc_denom)
          }
          pos += 1
        }
        llhs += llhs_th
        wllhs += wllhs_th
        dllhs += dllhs_th
      }))
      Await.ready(all, 2.hour)
      es.shutdown()

      (llhs, wllhs, dllhs)
    })))

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

  private def calcWSparseSum(counter: TC,
    alphaK_denoms: BDV[Double],
    numTopics: Int): Double = {
    var wSparseSum = 0D
    counter match {
      case v: BDV[Count] =>
        var i = 0
        while (i < numTopics) {
          wSparseSum += v(i) * alphaK_denoms(i)
          i += 1
        }
      case v: BSV[Count] =>
        val used = v.used
        val index = v.index
        val data = v.data
        var i = 0
        while (i < used) {
          wSparseSum += data(i) * alphaK_denoms(index(i))
          i += 1
        }
    }
    wSparseSum
  }

  private def calcDwSum(docTopics: BSV[Count],
    termBeta_denoms: BDV[Double]) = {
    // \frac{{n}_{kw}{n}_{kd}}{{n}_{k}+\bar{\beta}}
    val used = docTopics.used
    val index = docTopics.index
    val data = docTopics.data
    var dwSum = 0D
    var i = 0
    while (i < used) {
      dwSum += data(i) * termBeta_denoms(index(i))
      i += 1
    }
    dwSum
  }

  private def calcTermBetaDenoms(orgTermTopics: BV[Count],
    beta_denoms: BDV[Double],
    denoms: BDV[Double],
    numTopics: Int): BDV[Double] = {
    val bdv = beta_denoms.copy
    orgTermTopics match {
      case v: BDV[Count] =>
        var i = 0
        while (i < numTopics) {
          bdv(i) += v(i) * denoms(i)
          i += 1
        }
      case v: BSV[Count] =>
        val used = v.used
        val index = v.index
        val data = v.data
        var i = 0
        while (i < used) {
          val topic = index(i)
          bdv(topic) += data(i) * denoms(topic)
          i += 1
        }
    }
    bdv
  }
}
