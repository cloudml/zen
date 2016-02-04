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

import java.util.Random
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicIntegerArray

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, convert, sum}
import breeze.numerics._
import com.github.cloudml.zen.ml.clustering.LDADefines._
import com.github.cloudml.zen.ml.sampler.{AliasTable, CumulativeDist, DiscreteSampler, FTree}
import com.github.cloudml.zen.ml.util.Concurrent._
import com.github.cloudml.zen.ml.util.{BVCompressor, BVDecompressor, XORShiftRandom}
import org.apache.spark.graphx2.impl.{EdgePartition, ShippableVertexPartition => VertPartition}

import scala.collection.JavaConversions._
import scala.concurrent._


class LDAInferrer(numTopics: Int, numThreads: Int)
  extends LDAAlgorithm(numTopics, numThreads) {
  override def isByDoc: Boolean = false

  override def samplePartition(numPartitions: Int,
    sampIter: Int,
    seed: Int,
    topicCounters: BDV[Count],
    numTokens: Long,
    numTerms: Int,
    alpha: Double,
    alphaAS: Double,
    beta: Double)
    (pid: Int, ep: EdgePartition[TA, Nvk]): EdgePartition[TA, Int] = {
    val alphaRatio = alpha * numTopics / (numTokens + alphaAS * numTopics)
    val betaSum = beta * numTerms
    val denoms = calc_denoms(topicCounters, betaSum)
    val alphak_denoms = calc_alphak_denoms(denoms, alphaAS, betaSum, alphaRatio)
    val beta_denoms = denoms.copy :*= beta
    val totalSize = ep.size
    val lcSrcIds = ep.localSrcIds
    val lcDstIds = ep.localDstIds
    val vattrs = ep.vertexAttrs
    val data = ep.data
    val activeLens = new Array[Int](vattrs.length)
    val thq = new ConcurrentLinkedQueue(1 to numThreads)
    // table/ftree is a per term data structure
    // in GraphX, edges in a partition are clustered by source IDs (term id in this case)
    // so, use below simple cache to avoid calculating table each time
    val global: DiscreteSampler[Double] = new AliasTable
    val gens = new Array[XORShiftRandom](numThreads)
    val termDists = new Array[DiscreteSampler[Double]](numThreads)
    val cdfDists = new Array[CumulativeDist[Double]](numThreads)
    resetDist_abDense(global, alphak_denoms, beta)

    implicit val es = initExecutionContext(numThreads)
    val all = Future.traverse(ep.index.iterator)(Function.tupled((_, offset) => withFuture {
      val thid = thq.poll() - 1
      try {
        var gen = gens(thid)
        if (gen == null) {
          gen = new XORShiftRandom(((seed + sampIter) * numPartitions + pid) * numThreads + thid)
          gens(thid) = gen
          termDists(thid) = new AliasTable[Double].reset(numTopics)
          cdfDists(thid) = new CumulativeDist[Double].reset(numTopics)
        }
        val termDist = termDists(thid)
        val si = lcSrcIds(offset)
        val termTopics = vattrs(si)
        activeLens(si) = termTopics.activeSize
        resetDist_waSparse(termDist, alphak_denoms, termTopics)
        val denseTermTopics = toBDV(termTopics)
        val termBeta_denoms = calc_termBeta_denoms(denoms, beta_denoms, termTopics)
        val cdfDist = cdfDists(thid)
        var pos = offset
        while (pos < totalSize && lcSrcIds(pos) == si) {
          val di = lcDstIds(pos)
          val docTopics = vattrs(di).asInstanceOf[BSV[Count]]
          val topic = data(pos)
          resetDist_dwbSparse_withAdjust(cdfDist, denoms, termBeta_denoms, docTopics, topic)
          data(pos) = tokenSampling(gen, global, termDist, cdfDist, denseTermTopics, topic)
          pos += 1
        }
      } finally {
        thq.add(thid + 1)
      }
    }))
    withAwaitReadyAndClose(all)

    ep.withVertexAttributes(activeLens)
  }

  def tokenSampling(gen: Random,
    ab: DiscreteSampler[Double],
    wa: DiscreteSampler[Double],
    dwb: CumulativeDist[Double],
    termTopics: BDV[Count],
    topic: Int): Int = {
    val dwbSum = dwb.norm
    val sum23 = dwbSum + wa.norm
    val distSum = sum23 + ab.norm
    val genSum = gen.nextDouble() * distSum
    if (genSum < dwbSum) {
      dwb.sampleFrom(genSum, gen)
    } else if (genSum < sum23) wa match {
      case wt: AliasTable[Double] =>
        val rr = 1.0 / termTopics(topic)
        wt.resampleFrom(genSum - dwbSum, gen, topic, rr)
      case wf: FTree[Double] => wf.sampleFrom(genSum - dwbSum, gen)
    } else {
      ab.sampleFrom(genSum - sum23, gen)
    }
  }

  def tokenResampling(gen: Random,
    ab: DiscreteSampler[Double],
    wa: DiscreteSampler[Double],
    dwb: CumulativeDist[Double],
    termTopics: BDV[Count],
    docTopics: Ndk,
    topic: Int,
    beta: Double): Int = {
    val dwbSum = dwb.norm
    val sum23 = dwbSum + wa.norm
    val distSum = sum23 + ab.norm
    val genSum = gen.nextDouble() * distSum
    if (genSum < dwbSum) {
      val nkd = docTopics(topic)
      val nkw_beta = termTopics(topic) + beta
      val rr = 1.0 / nkd + 1.0 / nkw_beta - 1.0 / nkd / nkw_beta
      dwb.resampleFrom(genSum, gen, topic, rr)
    } else if (genSum < sum23) wa match {
      case wt: AliasTable[Double] =>
        val rr = 1.0 / termTopics(topic)
        wt.resampleFrom(genSum - dwbSum, gen, topic, rr)
      case wf: FTree[Double] => wf.sampleFrom(genSum - dwbSum, gen)
    } else {
      ab.sampleFrom(genSum - sum23, gen)
    }
  }

  override def countPartition(ep: EdgePartition[TA, Int]): Iterator[NvkPair] = {
    val totalSize = ep.size
    val lcSrcIds = ep.localSrcIds
    val lcDstIds = ep.localDstIds
    val l2g = ep.local2global
    val vattrs = ep.vertexAttrs
    val data = ep.data
    val vertSize = vattrs.length
    val results = new Array[NvkPair](vertSize)
    val marks = new AtomicIntegerArray(vertSize)

    implicit val es = initExecutionContext(numThreads)
    val all = Future.traverse(ep.index.iterator)(Function.tupled((_, offset) => withFuture {
      val si = lcSrcIds(offset)
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
        docTopics.synchronized {
          docTopics(topic) += 1
        }
        pos += 1
      }
    }))
    withAwaitReadyAndClose(all)

    results.iterator.filter(_ != null)
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
    val docNorms = new Array[Double](vertSize)
    val marks = new AtomicIntegerArray(vertSize)
    @volatile var llhs = 0.0
    @volatile var wllhs = 0.0
    @volatile var dllhs = 0.0

    implicit val es = initExecutionContext(numThreads)
    val all = Future.traverse(ep.index.iterator)(Function.tupled((_, offset) => withFuture {
      val si = lcSrcIds(offset)
      val termTopics = vattrs(si)
      val waSparseSum = sum_waSparse(alphak_denoms, termTopics)
      val sum12 = abDenseSum + waSparseSum
      val termBeta_denoms = calc_termBeta_denoms(denoms, beta_denoms, termTopics)
      var llhs_th = 0.0
      var wllhs_th = 0.0
      var dllhs_th = 0.0
      var pos = offset
      while (pos < totalSize && lcSrcIds(pos) == si) {
        val di = lcDstIds(pos)
        val docTopics = vattrs(di).asInstanceOf[BSV[Count]]
        if (marks.get(di) == 0) {
          docNorms(di) = 1.0 / (sum(docTopics) + alphaSum)
          marks.set(di, 1)
        }
        val docNorm = docNorms(di)
        val topic = data(pos)
        val dwbSparseSum = sum_dwbSparse(termBeta_denoms, docTopics)
        val prob = (sum12 + dwbSparseSum) * docNorm
        llhs_th += Math.log(prob)
        wllhs_th += Math.log(termBeta_denoms(topic))
        dllhs_th += Math.log((docTopics(topic) + alphaks(topic)) * docNorm)
        pos += 1
      }
      llhs += llhs_th
      wllhs += wllhs_th
      dllhs += dllhs_th
    }))
    withAwaitReadyAndClose(all)

    (llhs, wllhs, dllhs)
  }

  override def logLikelihoodPartition(topicCounters: BDV[Count],
    numTokens: Long,
    alpha: Double,
    beta: Double,
    alphaAS: Double)
    (vp: VertPartition[TC]): (Double, Double) = {
    val alphaSum = alpha * numTopics
    val alphaRatio = calc_alphaRatio(alphaSum, numTokens, alphaAS)
    val alphaks = calc_alphaks(topicCounters, alphaAS, alphaRatio)
    val lgamma_alphaks = alphaks.map(lgamma(_))
    val lgamma_beta = lgamma(beta)

    val totalSize = vp.capacity
    val index = vp.index
    val mask = vp.mask
    val values = vp.values
    @volatile var wllhs = 0.0
    @volatile var dllhs = 0.0
    val sizePerthrd = {
      val npt = totalSize / numThreads
      if (npt * numThreads == totalSize) npt else npt + 1
    }

    implicit val es = initExecutionContext(numThreads)
    val all = Future.traverse(Range(0, numThreads).iterator)(thid => withFuture {
      val decomp = new BVDecompressor(numTopics)
      val startPos = sizePerthrd * thid
      val endPos = math.min(sizePerthrd * (thid + 1), totalSize)
      var wllh_th = 0.0
      var dllh_th = 0.0
      var pos = mask.nextSetBit(startPos)
      while (pos < endPos && pos >= 0) {
        val bv = decomp.CV2BV(values(pos))
        if (isTermId(index.getValue(pos))) {
          bv.activeValuesIterator.foreach(cnt =>
            wllh_th += lgamma(cnt + beta)
          )
          wllh_th -= bv.activeSize * lgamma_beta
        } else {
          val docTopics = bv.asInstanceOf[BSV[Count]]
          val used = docTopics.used
          val index = docTopics.index
          val data = docTopics.data
          var nd = 0
          var i = 0
          while (i < used) {
            val topic = index(i)
            val cnt = data(i)
            dllh_th += lgamma(cnt + alphaks(topic)) - lgamma_alphaks(topic)
            nd += cnt
            i += 1
          }
          dllh_th -= lgamma(nd + alphaSum)
        }
        pos = mask.nextSetBit(pos + 1)
      }
      wllhs += wllh_th
      dllhs += dllh_th
    })
    withAwaitReadyAndClose(all)

    (wllhs, dllhs)
  }

  override def aggregateCounters(vp: VertPartition[TC],
    cntsIter: Iterator[NvkPair]): VertPartition[TC] = {
    val totalSize = vp.capacity
    val index = vp.index
    val mask = vp.mask
    val values = vp.values
    val results = new Array[BSV[Count]](totalSize)
    val marks = new AtomicIntegerArray(totalSize)

    implicit val es = initExecutionContext(numThreads)
    val all = Future.traverse(cntsIter.grouped(numThreads * 5).toIterator)(batch => Future {
      batch.foreach { case (vid, counter) =>
        assert(isDocId(vid))
        val bsv = counter.asInstanceOf[BSV[Count]]
        val i = index.getPos(vid)
        if (marks.getAndDecrement(i) == 0) {
          results(i) = bsv
        } else {
          while (marks.getAndSet(i, -1) <= 0) {}
          results(i) :+= bsv
        }
        marks.set(i, Int.MaxValue)
      }
    })
    withAwaitReady(all)

    // compress counters
    val sizePerthrd = {
      val npt = totalSize / numThreads
      if (npt * numThreads == totalSize) npt else npt + 1
    }
    val all2 = Future.traverse(Range(0, numThreads).iterator)(thid => withFuture {
      val comp = new BVCompressor(numTopics)
      val startPos = sizePerthrd * thid
      val endPos = math.min(sizePerthrd * (thid + 1), totalSize)
      var pos = mask.nextSetBit(startPos)
      while (pos < endPos && pos >= 0) {
        values(pos) = comp.BV2CV(results(pos))
        pos = mask.nextSetBit(pos + 1)
      }
    })
    withAwaitReadyAndClose(all2)

    vp.withValues(values)
  }

  def resetDist_abDense(ab: DiscreteSampler[Double],
    alphak_denoms: BDV[Double],
    beta: Double): DiscreteSampler[Double] = {
    val probs = alphak_denoms.copy :*= beta
    ab.resetDist(probs.data, null, probs.length)
  }

  def sum_abDense(alphak_denoms: BDV[Double],
    beta: Double): Double = {
    sum(alphak_denoms.copy :*= beta)
  }

  @inline def calc_alphaRatio(alphaSum: Double, numTokens: Long, alphaAS: Double): Double = {
    alphaSum / (numTokens + alphaAS * numTopics)
  }

  def calc_denoms(topicCounters: BDV[Count],
    betaSum: Double): BDV[Double] = {
    val arr = new Array[Double](numTopics)
    var i = 0
    while (i < numTopics) {
      arr(i) = 1.0 / (topicCounters(i) + betaSum)
      i += 1
    }
    new BDV(arr)
  }

  def calc_alphak_denoms(denoms: BDV[Double],
    alphaAS: Double,
    betaSum: Double,
    alphaRatio: Double): BDV[Double] = {
    (denoms.copy :*= ((alphaAS - betaSum) * alphaRatio)) :+= alphaRatio
  }

  def calc_beta_denoms(denoms: BDV[Double],
    beta: Double): BDV[Double] = {
    denoms.copy :*= beta
  }

  def calc_alphaks(topicCounters: BDV[Count],
    alphaAS: Double,
    alphaRatio: Double): BDV[Double] = {
    (convert(topicCounters, Double) :+= alphaAS) :*= alphaRatio
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
}
