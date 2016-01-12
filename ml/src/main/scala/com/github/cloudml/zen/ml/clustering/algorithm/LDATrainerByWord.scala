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

import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicIntegerArray

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, convert, sum}
import com.github.cloudml.zen.ml.clustering.LDADefines._
import com.github.cloudml.zen.ml.sampler._
import org.apache.spark.graphx2.impl.EdgePartition

import scala.concurrent._
import scala.concurrent.duration._


abstract class LDATrainerByWord(numTopics: Int, numThreads: Int)
  extends LDATrainer(numTopics, numThreads) {
  override def isByDoc: Boolean = false

  override def countPartition(ep: EdgePartition[TA, Int]): Iterator[NvkPair] = {
    val totalSize = ep.size
    val lcSrcIds = ep.localSrcIds
    val lcDstIds = ep.localDstIds
    val l2g = ep.local2global
    val activeLens = ep.vertexAttrs
    val data = ep.data
    val vertSize = activeLens.length
    val results = new Array[NvkPair](vertSize)
    val marks = new AtomicIntegerArray(vertSize)

    implicit val es = ExecutionContext.fromExecutorService(Executors.newFixedThreadPool(numThreads))
    val all = Future.traverse(ep.index.iterator)(Function.tupled((_, offset) => Future {
      val si = lcSrcIds(offset)
      var termTuple = results(si)
      if (termTuple == null) {
        termTuple = (l2g(si), BSV.zeros[Count](numTopics))
        results(si) = termTuple
      }
      var termTopics = termTuple._2
      var pos = offset
      while (pos < totalSize && lcSrcIds(pos) == si) {
        val di = lcDstIds(pos)
        var docTuple = results(di)
        if (docTuple == null) {
          if (marks.getAndDecrement(di) == 0) {
            docTuple = (l2g(di), BSV.zeros[Count](numTopics))
            results(di) = docTuple
            marks.set(di, Int.MaxValue)
          } else {
            while (marks.get(di) <= 0) {}
            docTuple = results(di)
          }
        }
        val docTopics = docTuple._2
        val topic = data(pos)
        termTopics match {
          case v: BDV[Count] => v(topic) += 1
          case v: BSV[Count] =>
            v(topic) += 1
            if (v.activeSize >= dscp) {
              termTuple = (l2g(si), toBDV(v))
              results(si) = termTuple
              termTopics = termTuple._2
            }
        }
        docTopics.synchronized {
          docTopics(topic) += 1
        }
        pos += 1
      }
    }))
    Await.ready(all, 1.hour)
    es.shutdown()
    results.iterator
  }

  override def perplexPartition(topicCounters: BDV[Count],
    numTokens: Long,
    numTerms: Int,
    alpha: Double,
    alphaAS: Double,
    beta: Double)
    (ep: EdgePartition[TA, Nvk]): (Double, Double, Double) = {
    val alphaSum = alpha * numTopics
    val betaSum = beta * numTerms
    val alphaRatio = alphaSum / (numTokens + alphaAS * numTopics)
    val alphaks = (convert(topicCounters, Double) :+= alphaAS) :*= alphaRatio
    val denoms = calc_denoms(topicCounters, betaSum)
    val alphak_denoms = calc_alphak_denoms(denoms, alphaAS, betaSum, alphaRatio)
    val beta_denoms = denoms.copy :*= beta
    // \frac{{\alpha }_{k}{\beta }_{w}}{{n}_{k}+\bar{\beta }}
    val abDenseSum = sum_abDense(alphak_denoms, beta)
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
      val termTopics = vattrs(si)
      val waSparseSum = sum_waSparse(alphak_denoms, termTopics)
      val sum12 = abDenseSum + waSparseSum
      val termBeta_denoms = calc_termBeta_denoms(denoms, beta_denoms, termTopics)
      var llhs_th = 0D
      var wllhs_th = 0D
      var dllhs_th = 0D
      var pos = offset
      while (pos < totalSize && lcSrcIds(pos) == si) {
        val di = lcDstIds(pos)
        val docTopics = vattrs(di).asInstanceOf[BSV[Count]]
        if (marks.get(di) == 0) {
          doc_denoms(di) = 1.0 / (sum(docTopics) + alphaSum)
          marks.set(di, 1)
        }
        val doc_denom = doc_denoms(di)
        val topic = data(pos)
        val dwbSparseSum = sum_dwbSparse(termBeta_denoms, docTopics)
        val prob = (sum12 + dwbSparseSum) * doc_denom
        llhs_th += Math.log(prob)
        wllhs_th += Math.log(termBeta_denoms(topic))
        dllhs_th += Math.log((docTopics(topic) + alphaks(topic)) * doc_denom)
        pos += 1
      }
      llhs += llhs_th
      wllhs += wllhs_th
      dllhs += dllhs_th
    }))
    Await.ready(all, 2.hour)
    es.shutdown()
    (llhs, wllhs, dllhs)
  }

  def resetDist_waSparse(wa: DiscreteSampler[Double],
    alphak_denoms: BDV[Double],
    termTopics: Nwk): DiscreteSampler[Double] = termTopics match {
    case v: BDV[Count] =>
      val k = v.length
      val probs = new Array[Double](k)
      val space = new Array[Int](k)
      var psize = 0
      var i = 0
      while (i < k) {
        val cnt = v(i)
        if (cnt > 0) {
          probs(psize) = alphak_denoms(i) * cnt
          space(psize) = i
          psize += 1
        }
        i += 1
      }
      wa.resetDist(probs, space, psize)
    case v: BSV[Count] =>
      val used = v.used
      val index = v.index
      val data = v.data
      val probs = new Array[Double](used)
      var i = 0
      while (i < used) {
        probs(i) = alphak_denoms(index(i)) * data(i)
        i += 1
      }
      wa.resetDist(probs, index, used)
  }

  def sum_waSparse(alphak_denoms: BDV[Double],
    termTopics: Nwk): Double = termTopics match {
    case v: BDV[Count] =>
      val k = v.length
      var sum = 0.0
      var i = 0
      while (i < k) {
        val cnt = v(i)
        if (cnt > 0) {
          sum += alphak_denoms(i) * cnt
        }
        i += 1
      }
      sum
    case v: BSV[Count] =>
      val used = v.used
      val index = v.index
      val data = v.data
      var sum = 0.0
      var i = 0
      while (i < used) {
        sum += alphak_denoms(index(i)) * data(i)
        i += 1
      }
      sum
  }

  def resetDist_dwbSparse(dwb: CumulativeDist[Double],
    termBeta_denoms: BDV[Double],
    docTopics: Ndk): CumulativeDist[Double] = {
    val used = docTopics.used
    val index = docTopics.index
    val data = docTopics.data
    // DANGER operations for performance
    dwb._used = used
    val cdf = dwb._cdf
    var sum = 0.0
    var i = 0
    while (i < used) {
      sum += termBeta_denoms(index(i)) * data(i)
      cdf(i) = sum
      i += 1
    }
    dwb._space = index
    dwb
  }

  def resetDist_dwbSparse_withAdjust(dwb: CumulativeDist[Double],
    denoms: BDV[Double],
    termBeta_denoms: BDV[Double],
    docTopics: Ndk,
    curTopic: Int): CumulativeDist[Double] = {
    val used = docTopics.used
    val index = docTopics.index
    val data = docTopics.data
    // DANGER operations for performance
    dwb._used = used
    val cdf = dwb._cdf
    var sum = 0.0
    var i = 0
    while (i < used) {
      val topic = index(i)
      val prob = if (topic == curTopic) {
        (termBeta_denoms(topic) - denoms(topic)) * (data(i) - 1)
      } else {
        termBeta_denoms(topic) * data(i)
      }
      sum += prob
      cdf(i) = sum
      i += 1
    }
    dwb._space = index
    dwb
  }

  def sum_dwbSparse(termBeta_denoms: BDV[Double],
    docTopics: Ndk): Double = {
    val used = docTopics.used
    val index = docTopics.index
    val data = docTopics.data
    var sum = 0.0
    var i = 0
    while (i < used) {
      sum += termBeta_denoms(index(i)) * data(i)
      i += 1
    }
    sum
  }

  def calc_termBeta_denoms(denoms: BDV[Double],
    beta_denoms: BDV[Double],
    termTopics: Nwk): BDV[Double] = {
    val bdv = beta_denoms.copy
    termTopics match {
      case v: BDV[Count] =>
        val k = v.length
        var i = 0
        while (i < k) {
          bdv(i) += denoms(i) * v(i)
          i += 1
        }
      case v: BSV[Count] =>
        val used = v.used
        val index = v.index
        val data = v.data
        var i = 0
        while (i < used) {
          val topic = index(i)
          bdv(topic) += denoms(topic) * data(i)
          i += 1
        }
    }
    bdv
  }
}
