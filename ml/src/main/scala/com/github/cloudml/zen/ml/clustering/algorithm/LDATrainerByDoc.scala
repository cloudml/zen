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

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, sum}
import com.github.cloudml.zen.ml.clustering.LDADefines._
import com.github.cloudml.zen.ml.sampler._
import com.github.cloudml.zen.ml.util.Concurrent._
import org.apache.spark.graphx2.impl.EdgePartition

import scala.concurrent._


abstract class LDATrainerByDoc(numTopics: Int, numThreads: Int)
  extends LDATrainer(numTopics: Int, numThreads: Int) {
  override def isByDoc: Boolean = true

  override def countPartition(ep: EdgePartition[TA, Int]): Iterator[NvkPair] = {
    val totalSize = ep.size
    val lcSrcIds = ep.localSrcIds
    val lcDstIds = ep.localDstIds
    val l2g = ep.local2global
    val useds = ep.vertexAttrs
    val data = ep.data
    val vertSize = useds.length
    val results = new Array[NvkPair](vertSize)

    implicit val es = initExecutionContext(numThreads)
    val all0 = Future.traverse(Range(0, numThreads).iterator) { thid => withFuture {
      var i = thid
      while (i < vertSize) {
        val vid = l2g(i)
        val used = useds(i)
        val counter: Nvk = if (isTermId(vid) && used >= dscp) {
          new BDV(new Array[Count](numTopics))
        } else {
          val len = math.max(used >>> 1, 2)
          new BSV(new Array[Int](len), new Array[Count](len), 0, numTopics)
        }
        results(i) = (vid, counter)
        i += numThreads
      }
    }}
    withAwaitReady(all0)

    val all = Future.traverse(ep.index.iterator) { case (_, startPos) => withFuture {
      val si = lcSrcIds(startPos)
      val docTopics = results(si)._2
      var pos = startPos
      while (pos < totalSize && lcSrcIds(pos) == si) {
        val termTopics = results(lcDstIds(pos))._2
        val topic = data(pos)
        docTopics(topic) += 1
        termTopics.synchronized {
          termTopics(topic) += 1
        }
        pos += 1
      }
    }}
    withAwaitReady(all)

    val all2 = Future.traverse(Range(0, numThreads).iterator) { thid => withFuture {
      var i = thid
      while (i < vertSize) {
        val tuple = results(i)
        val vid = tuple._1
        if (isTermId(vid)) tuple._2 match {
          case v: BDV[Count] =>
            val used = v.data.count(_ > 0)
            if (used < dscp) {
              results(i) = (vid, toBSV(v, used))
            }
          case _ =>
        }
        i += numThreads
      }
    }}
    withAwaitReadyAndClose(all2)

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
    val alphaRatio = calc_alphaRatio(alphaSum, numTokens, alphaAS)
    val alphaks = calc_alphaks(topicCounters, alphaAS, alphaRatio)
    val denoms = calc_denoms(topicCounters, betaSum)
    val alphak_denoms = calc_alphak_denoms(denoms, alphaAS, betaSum, alphaRatio)

    val totalSize = ep.size
    val lcSrcIds = ep.localSrcIds
    val lcDstIds = ep.localDstIds
    val vattrs = ep.vertexAttrs
    val data = ep.data
    @volatile var llhs = 0D
    @volatile var wllhs = 0D
    @volatile var dllhs = 0D
    val abDenseSum = sum_abDense(alphak_denoms, beta)

    implicit val es = initExecutionContext(numThreads)
    val all = Future.traverse(ep.index.iterator)(Function.tupled((_, offset) => withFuture {
      val si = lcSrcIds(offset)
      val docTopics = vattrs(si).asInstanceOf[BSV[Count]]
      val doc_denom = 1.0 / (sum(docTopics) + alphaSum)
      val nkd_denoms = calc_nkd_denoms(denoms, docTopics)
      val dbSparseSum = sum_dbSparse(nkd_denoms, beta)
      val sum12 = abDenseSum + dbSparseSum
      val docAlphaK_denoms = calc_docAlphaK_denoms(alphak_denoms, nkd_denoms)
      var llhs_th = 0D
      var wllhs_th = 0D
      var dllhs_th = 0D
      var pos = offset
      while (pos < totalSize && lcSrcIds(pos) == si) {
        val di = lcDstIds(pos)
        val termTopics = vattrs(di)
        val topic = data(pos)
        val wdaSparseSum = sum_wdaSparse(docAlphaK_denoms, termTopics)
        val prob = (sum12 + wdaSparseSum) * doc_denom
        llhs_th += Math.log(prob)
        wllhs_th += Math.log((termTopics(topic) + beta) * denoms(topic))
        dllhs_th += Math.log((docTopics(topic) + alphaks(topic)) * doc_denom)
        pos += 1
      }
      llhs += llhs_th
      wllhs += wllhs_th
      dllhs += dllhs_th
    }))
    withAwaitReadyAndClose(all)

    (llhs, wllhs, dllhs)
  }

  def resetDist_dbSparse(db: FlatDist[Double],
    nkd_denoms: BSV[Double],
    beta: Double): FlatDist[Double] = {
    val used = nkd_denoms.used
    val index = nkd_denoms.index
    val data = nkd_denoms.data
    val probs = new Array[Double](used)
    var i = 0
    while (i < used) {
      probs(i) = beta * data(i)
      i += 1
    }
    db.resetDist(probs, index, used)
  }

  def sum_dbSparse(nkd_denoms: BSV[Double],
    beta: Double): Double = {
    val used = nkd_denoms.used
    val data = nkd_denoms.data
    var sum = 0.0
    var i = 0
    while (i < used) {
      sum += beta * data(i)
      i += 1
    }
    sum
  }

  def resetDist_wdaSparse(wda: FlatDist[Double],
    docAlphaK_Denoms: BDV[Double],
    termTopics: Nwk): FlatDist[Double] = termTopics match {
    case v: BDV[Count] =>
      val probs = new Array[Double](numTopics)
      val space = new Array[Int](numTopics)
      var psize = 0
      var i = 0
      while (i < numTopics) {
        val cnt = v(i)
        if (cnt > 0) {
          probs(psize) = docAlphaK_Denoms(i) * cnt
          space(psize) = i
          psize += 1
        }
        i += 1
      }
      wda.resetDist(probs, space, psize)
    case v: BSV[Count] =>
      val used = v.used
      val index = v.index
      val data = v.data
      val probs = new Array[Double](used)
      var i = 0
      while (i < used) {
        probs(i) = docAlphaK_Denoms(index(i)) * data(i)
        i += 1
      }
      wda.resetDist(probs, index, used)
  }

  def sum_wdaSparse(docAlphaK_Denoms: BDV[Double],
    termTopics: Nwk): Double = termTopics match {
    case v: BDV[Count] =>
      var sum = 0.0
      var i = 0
      while (i < numTopics) {
        val cnt = v(i)
        if (cnt > 0) {
          sum += docAlphaK_Denoms(i) * cnt
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
        sum += docAlphaK_Denoms(index(i)) * data(i)
        i += 1
      }
      sum
  }

  def calc_nkd_denoms(denoms: BDV[Double],
    docTopics: Ndk): BSV[Double] = {
    val used = docTopics.used
    val index = docTopics.index
    val data = docTopics.data
    val arr = new Array[Double](used)
    var i = 0
    while (i < used) {
      arr(i) = data(i) * denoms(index(i))
      i += 1
    }
    new BSV(index, arr, used, denoms.length)
  }

  def calc_docAlphaK_denoms(alphak_denoms: BDV[Double],
    nkd_denoms: BSV[Double]): BDV[Double] = {
    val bdv = alphak_denoms.copy
    val used = nkd_denoms.used
    val index = nkd_denoms.index
    val data = nkd_denoms.data
    var i = 0
    while (i < used) {
      bdv(index(i)) += data(i)
      i += 1
    }
    bdv
  }
}
