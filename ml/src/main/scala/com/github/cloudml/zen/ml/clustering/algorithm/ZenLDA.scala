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
import java.util.concurrent.{ConcurrentLinkedQueue, Executors}

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, sum}
import com.github.cloudml.zen.ml.clustering.LDADefines._
import com.github.cloudml.zen.ml.sampler._
import com.github.cloudml.zen.ml.util.XORShiftRandom
import org.apache.spark.graphx2.impl.EdgePartition

import scala.collection.JavaConversions._
import scala.concurrent._
import scala.concurrent.duration._


class ZenLDA(numTopics: Int, numThreads: Int)
  extends LDATrainerByWord(numTopics, numThreads) {
  override def initEdgePartition(ep: EdgePartition[TA, _]): EdgePartition[TA, Int] = {
    val ep2 = super.initEdgePartition(ep)
    val lcSrcIds = ep2.localSrcIds
    val lcDstIds = ep2.localDstIds
    implicit val es = ExecutionContext.fromExecutorService(Executors.newFixedThreadPool(numThreads))
    val all = Future.traverse(ep2.index.iterator.zipWithIndex) { case ((_, startPos), ii) => Future {
      val endPos = lcSrcIds(ii << 1 + 1)
      var anchor = startPos
      var anchorId = lcDstIds(anchor)
      var pos = startPos + 1
      while (pos < endPos) {
        val lcDstId = lcDstIds(pos)
        if (lcDstId != anchorId) {
          val numLink = pos - anchor
          if (numLink > 1) {
            lcDstIds(anchor) = -numLink
          }
          anchor = pos
          anchorId = lcDstId
        }
        pos += 1
      }
      val numLink = pos - anchor
      if (numLink > 1) {
        lcDstIds(anchor) = -numLink
      }
    }}
    Await.ready(all, 1.hour)
    es.shutdown()
    new EdgePartition(lcSrcIds, lcDstIds, ep2.data, ep2.index, ep2.global2local, ep2.local2global,
      ep2.vertexAttrs, None)
  }

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
    val alphaSum = alpha * numTopics
    val betaSum = beta * numTerms
    val alphaRatio = calc_alphaRatio(alphaSum, numTokens, alphaAS)
    val denoms = calc_denoms(topicCounters, betaSum)
    val alphak_denoms = calc_alphak_denoms(denoms, alphaAS, betaSum, alphaRatio)
    val beta_denoms = calc_beta_denoms(denoms, beta)

    val lcSrcIds = ep.localSrcIds
    val lcDstIds = ep.localDstIds
    val vattrs = ep.vertexAttrs
    val data = ep.data
    val useds = new Array[Int](vattrs.length)
    val thq = new ConcurrentLinkedQueue(0 until numThreads)
    // table/ftree is a per term data structure
    // in GraphX, edges in a partition are clustered by source IDs (term id in this case)
    // so, use below simple cache to avoid calculating table each time
    val global: DiscreteSampler[Double] = new AliasTable
    val gens = new Array[XORShiftRandom](numThreads)
    val termDists = new Array[DiscreteSampler[Double]](numThreads)
    val cdfDists = new Array[CumulativeDist[Double]](numThreads)
    resetDist_abDense(global, alphak_denoms, beta)

    implicit val es = ExecutionContext.fromExecutorService(Executors.newFixedThreadPool(numThreads))
    val all = Future.traverse(ep.index.iterator.zipWithIndex) { case ((_, startPos), ii) => Future {
      val thid = thq.poll()
      var gen = gens(thid)
      if (gen == null) {
        gen = new XORShiftRandom(((seed + sampIter) * numPartitions + pid) * numThreads + thid)
        gens(thid) = gen
        termDists(thid) = new AliasTable[Double] { reset(numTopics) }
        cdfDists(thid) = new CumulativeDist[Double] { reset(numTopics) }
      }
      val termDist = termDists(thid)
      val cdfDist = cdfDists(thid)

      val lsi = ii << 1
      val si = lcSrcIds(lsi)
      val endPos = lcSrcIds(lsi + 1)
      val numSrcEdges = endPos - startPos
      val dlgPos = startPos + gen.nextInt(math.min(numSrcEdges, 32))
      val common = numSrcEdges * vattrs(lcDstIds(dlgPos)).activeSize >= dscp
      val termTopics = vattrs(si)
      useds(si) = termTopics.activeSize
      resetDist_waSparse(termDist, alphak_denoms, termTopics)
      val denseTermTopics = toBDV(termTopics)
      if (common) {
        val termBeta_denoms = calc_termBeta_denoms(denoms, beta_denoms, termTopics)
        var pos = startPos
        while (pos < endPos) {
          val ind = lcDstIds(pos)
          if (ind >= 0) {
            val di = ind
            val docTopics = vattrs(di).asInstanceOf[Ndk]
            useds(di) = docTopics.activeSize
            val topic = data(pos)
            resetDist_dwbSparse_wOptAdjust(cdfDist, denoms, termBeta_denoms, docTopics, topic)
            data(pos) = tokenSampling(gen, global, termDist, cdfDist, denseTermTopics, topic)
            pos += 1
          } else {
            val di = lcDstIds(pos + 1)
            val docTopics = vattrs(di).asInstanceOf[Ndk]
            useds(di) = docTopics.activeSize
            resetDist_dwbSparse_wOpt(cdfDist, termBeta_denoms, docTopics)
            var l = 0
            while (l > ind) {
              val topic = data(pos)
              data(pos) = tokenResampling(gen, global, termDist, cdfDist, denseTermTopics, docTopics, topic, beta)
              pos += 1
              l -= 1
            }
          }
        }
      } else {
        var pos = startPos
        while (pos < endPos) {
          val ind = lcDstIds(pos)
          if (ind >= 0) {
            val di = ind
            val docTopics = vattrs(di).asInstanceOf[Ndk]
            useds(di) = docTopics.activeSize
            val topic = data(pos)
            resetDist_dwbSparse_wAdjust(cdfDist, denoms, denseTermTopics, docTopics, topic, beta)
            data(pos) = tokenSampling(gen, global, termDist, cdfDist, denseTermTopics, topic)
            pos += 1
          } else {
            val di = lcDstIds(pos + 1)
            val docTopics = vattrs(di).asInstanceOf[Ndk]
            useds(di) = docTopics.activeSize
            resetDist_dwbSparse(cdfDist, denoms, denseTermTopics, docTopics, beta)
            var l = 0
            while (l > ind) {
              val topic = data(pos)
              data(pos) = tokenResampling(gen, global, termDist, cdfDist, denseTermTopics, docTopics, topic, beta)
              pos += 1
              l -= 1
            }
          }
        }
      }
      thq.add(thid)
    }}
    Await.ready(all, 2.hour)
    es.shutdown()
    ep.withVertexAttributes(useds)
  }

  def tokenSampling(gen: Random,
    ab: DiscreteSampler[Double],
    wa: DiscreteSampler[Double],
    dwb: CumulativeDist[Double],
    denseTermTopics: BDV[Count],
    topic: Int): Int = {
    val dwbSum = dwb.norm
    val sum23 = dwbSum + wa.norm
    val distSum = sum23 + ab.norm
    val genSum = gen.nextDouble() * distSum
    if (genSum < dwbSum) {
      dwb.sampleFrom(genSum, gen)
    } else if (genSum < sum23) {
      val rr = 1.0 / denseTermTopics(topic)
      wa.resampleFrom(genSum - dwbSum, gen, topic, rr)
    } else {
      ab.sampleFrom(genSum - sum23, gen)
    }
  }

  def tokenResampling(gen: Random,
    ab: DiscreteSampler[Double],
    wa: DiscreteSampler[Double],
    dwb: CumulativeDist[Double],
    denseTermTopics: BDV[Count],
    docTopics: Ndk,
    topic: Int,
    beta: Double): Int = {
    val dwbSum = dwb.norm
    val sum23 = dwbSum + wa.norm
    val distSum = sum23 + ab.norm
    val genSum = gen.nextDouble() * distSum
    if (genSum < dwbSum) {
      val a = 1.0 / (denseTermTopics(topic) + beta)
      val b = 1.0 / docTopics(topic)
      val rr = a + b - a * b
      dwb.resampleFrom(genSum, gen, topic, rr)
    } else if (genSum < sum23) {
      val rr = 1.0 / denseTermTopics(topic)
      wa.resampleFrom(genSum - dwbSum, gen, topic, rr)
    } else {
      ab.sampleFrom(genSum - sum23, gen)
    }
  }

  override def countPartition(ep: EdgePartition[TA, Int]): Iterator[NvkPair] = {
    val lcSrcIds = ep.localSrcIds
    val lcDstIds = ep.localDstIds
    val l2g = ep.local2global
    val useds = ep.vertexAttrs
    val data = ep.data
    val vertSize = useds.length
    val results = new Array[NvkPair](vertSize)

    implicit val es = ExecutionContext.fromExecutorService(Executors.newFixedThreadPool(numThreads))
    val all0 = Range(0, numThreads).map { thid => Future {
      var i = thid
      while (i < vertSize) {
        val used = useds(i)
        val counter: Nvk = if (used >= dscp) {
          new BDV(new Array[Count](numTopics))
        } else {
          val len = math.min(used >>> 1, 2)
          new BSV(new Array[Int](len), new Array[Count](len), 0, numTopics)
        }
        results(i) = (l2g(i), counter)
        i += numThreads
      }
    }}
    Await.ready(Future.sequence(all0), 1.hour)

    val all = Future.traverse(ep.index.iterator.zipWithIndex) { case ((_, startPos), ii) => Future {
      val lsi = ii << 1
      val si = lcSrcIds(lsi)
      val endPos = lcSrcIds(lsi + 1)
      val termTopics = results(si)._2
      var pos = startPos
      while (pos < endPos) {
        val ind = lcDstIds(pos)
        if (ind >= 0) {
          val di = ind
          val docTopics = results(di)._2
          val topic = data(pos)
          termTopics(topic) += 1
          docTopics.synchronized {
            docTopics(topic) += 1
          }
        } else {
          val di = lcDstIds(pos + 1)
          val docTopics = results(di)._2
          var l = 0
          while (l > ind) {
            val topic = data(pos)
            termTopics(topic) += 1
            docTopics.synchronized {
              docTopics(topic) += 1
            }
            pos += 1
            l -= 1
          }
        }
        pos += 1
      }
      termTopics match {
        case v: BDV[Count] =>
          val used = v.data.count(_ > 0)
          if (used < dscp) {
            results(si) = (l2g(si), toBSV(v, used))
          }
      }
    }}
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
    val alphaRatio = calc_alphaRatio(alphaSum, numTokens, alphaAS)
    val alphaks = calc_alphaks(topicCounters, alphaAS, alphaRatio)
    val denoms = calc_denoms(topicCounters, betaSum)
    val alphak_denoms = calc_alphak_denoms(denoms, alphaAS, betaSum, alphaRatio)
    val beta_denoms = calc_beta_denoms(denoms, beta)

    val lcSrcIds = ep.localSrcIds
    val lcDstIds = ep.localDstIds
    val vattrs = ep.vertexAttrs
    val data = ep.data
    val vertSize = vattrs.length
    val doc_denoms = new Array[Double](vertSize)
    @volatile var llhs = 0D
    @volatile var wllhs = 0D
    @volatile var dllhs = 0D
    val abDenseSum = sum_abDense(alphak_denoms, beta)

    implicit val es = ExecutionContext.fromExecutorService(Executors.newFixedThreadPool(numThreads))
    val all = Future.traverse(ep.index.iterator.zipWithIndex) { case ((_, startPos), ii) => Future {
      val lsi = ii << 1
      val si = lcSrcIds(lsi)
      val endPos = lcSrcIds(lsi + 1)
      val termTopics = vattrs(si)
      val waSparseSum = sum_waSparse(alphak_denoms, termTopics)
      val sum12 = abDenseSum + waSparseSum
      val termBeta_denoms = calc_termBeta_denoms(denoms, beta_denoms, termTopics)
      var llhs_th = 0D
      var wllhs_th = 0D
      var dllhs_th = 0D
      var pos = startPos
      while (pos < endPos) {
        val ind = lcDstIds(pos)
        if (ind >= 0) {
          val di = ind
          val docTopics = vattrs(di).asInstanceOf[Ndk]
          var doc_denom = doc_denoms(di)
          if (doc_denom == 0.0) {
            doc_denom = 1.0 / (sum(docTopics) + alphaSum)
            doc_denoms(di) = doc_denom
          }
          val topic = data(pos)
          val dwbSparseSum = sum_dwbSparse_wOpt(termBeta_denoms, docTopics)
          val prob = (sum12 + dwbSparseSum) * doc_denom
          llhs_th += Math.log(prob)
          wllhs_th += Math.log(termBeta_denoms(topic))
          dllhs_th += Math.log((docTopics(topic) + alphaks(topic)) * doc_denom)
          pos += 1
        } else {
          val di = lcDstIds(pos + 1)
          val docTopics = vattrs(di).asInstanceOf[Ndk]
          var doc_denom = doc_denoms(di)
          if (doc_denom == 0.0) {
            doc_denom = 1.0 / (sum(docTopics) + alphaSum)
            doc_denoms(di) = doc_denom
          }
          var l = 0
          while (l > ind) {
            val topic = data(pos)
            val dwbSparseSum = sum_dwbSparse_wOpt(termBeta_denoms, docTopics)
            val prob = (sum12 + dwbSparseSum) * doc_denom
            llhs_th += Math.log(prob)
            wllhs_th += Math.log(termBeta_denoms(topic))
            dllhs_th += Math.log((docTopics(topic) + alphaks(topic)) * doc_denom)
            pos += 1
            l -= 1
          }
        }
      }
      llhs += llhs_th
      wllhs += wllhs_th
      dllhs += dllhs_th
    }}
    Await.ready(all, 2.hour)
    es.shutdown()
    (llhs, wllhs, dllhs)
  }
}
