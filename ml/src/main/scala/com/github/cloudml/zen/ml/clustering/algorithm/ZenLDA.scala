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
    val totalSize = ep.size
    val srcSize = ep.indexSize
    val lcSrcIds = ep.localSrcIds
    val lcDstIds = ep.localDstIds
    val zeros = new Array[Int](ep.vertexAttrs.length)
    val srcInfos = new Array[(Int, Int, Int)](srcSize)

    implicit val es = initPartExecutionContext()
    val all = Future.traverse(ep.index.iterator.zipWithIndex) { case ((_, startPos), ii) => Future {
      val si = lcSrcIds(startPos)
      var anchor = startPos
      var anchorId = lcDstIds(anchor)
      var pos = startPos + 1
      while (pos < totalSize && lcSrcIds(pos) == si) {
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
      srcInfos(ii) = (si, startPos, pos)
    }}
    Await.ready(all, 1.hour)
    closePartExecutionContext()

    val newLcSrcIds = srcInfos.toSeq.sorted.flatMap(t => Iterator(t._1, t._2, t._3)).toArray
    new EdgePartition(newLcSrcIds, lcDstIds, ep.data, null, ep.global2local, ep.local2global, zeros, None)
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
    // table is a per term data structure
    // in GraphX, edges in a partition are clustered by source IDs (term id in this case)
    // so, use below simple cache to avoid calculating table each time
    val global: DiscreteSampler[Double] = new AliasTable
    val gens = new Array[XORShiftRandom](numThreads)
    val termDists = new Array[DiscreteSampler[Double]](numThreads)
    val cdfDists = new Array[CumulativeDist[Double]](numThreads)
    resetDist_abDense(global, alphak_denoms, beta)

    implicit val es = initPartExecutionContext()
    val all = Future.traverse(lcSrcIds.indices.by(3).iterator)(lsi => Future {
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

      val si = lcSrcIds(lsi)
      val startPos = lcSrcIds(lsi + 1)
      val endPos = lcSrcIds(lsi + 2)
      val termTopics = vattrs(si)
      useds(si) = termTopics.activeSize
      resetDist_waSparse(termDist, alphak_denoms, termTopics)
      val denseTermTopics = toBDV(termTopics)
      val common = isCommon(gen, startPos, endPos, lcDstIds, vattrs)
      var pos = startPos
      if (common) {
        val termBeta_denoms = calc_termBeta_denoms(denoms, beta_denoms, termTopics)
        while (pos < endPos) {
          var ind = lcDstIds(pos)
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
            while (ind < 0) {
              val topic = data(pos)
              data(pos) = tokenResampling(gen, global, termDist, cdfDist, denseTermTopics, docTopics, topic, beta)
              pos += 1
              ind += 1
            }
          }
        }
      } else {
        while (pos < endPos) {
          var ind = lcDstIds(pos)
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
            while (ind < 0) {
              val topic = data(pos)
              data(pos) = tokenResampling(gen, global, termDist, cdfDist, denseTermTopics, docTopics, topic, beta)
              pos += 1
              ind += 1
            }
          }
        }
      }
      thq.add(thid)
    })
    Await.ready(all, 2.hour)
    closePartExecutionContext()

    ep.withVertexAttributes(useds)
  }

  private def isCommon(gen: Random,
    startPos: Int,
    endPos: Int,
    lcDstIds: Array[Int],
    vattrs: Array[Nvk]): Boolean = {
    val numSrcEdges = endPos - startPos
    val dlgPos = startPos + gen.nextInt(math.min(numSrcEdges, 32))
    val dlgInd = lcDstIds(dlgPos)
    val dlgDi = if (dlgInd >= 0) dlgInd else lcDstIds(dlgPos + 1)
    numSrcEdges * vattrs(dlgDi).activeSize >= dscp
  }

  def tokenSampling(gen: Random,
    ab: DiscreteSampler[Double],
    wa: DiscreteSampler[Double],
    dwb: CumulativeDist[Double],
    denseTermTopics: BDV[Count],
    curTopic: Int): Int = {
    val dwbSum = dwb.norm
    val sum23 = dwbSum + wa.norm
    val distSum = sum23 + ab.norm
    val genSum = gen.nextDouble() * distSum
    if (genSum < dwbSum) {
      dwb.sampleFrom(genSum, gen)
    } else if (genSum < sum23) {
      val rr = 1.0 / denseTermTopics(curTopic)
      wa.resampleFrom(genSum - dwbSum, gen, curTopic, rr)
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
    curTopic: Int,
    beta: Double): Int = {
    val dwbSum = dwb.norm
    val sum23 = dwbSum + wa.norm
    val distSum = sum23 + ab.norm
    val genSum = gen.nextDouble() * distSum
    if (genSum < dwbSum) {
      val a = 1.0 / (denseTermTopics(curTopic) + beta)
      val b = 1.0 / docTopics(curTopic)
      val rr = a + b - a * b
      dwb.resampleFrom(genSum, gen, curTopic, rr)
    } else if (genSum < sum23) {
      val rr = 1.0 / denseTermTopics(curTopic)
      wa.resampleFrom(genSum - dwbSum, gen, curTopic, rr)
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

    implicit val es = initPartExecutionContext()
    val all0 = Future.traverse(Range(0, numThreads).iterator)(thid => Future {
      var i = thid
      while (i < vertSize) {
        val vid = l2g(i)
        val used = useds(i)
        val counter: Nvk = if (isTermId(vid) && used >= dscp) {
          new BDV(new Array[Count](numTopics))
        } else {
          val len = math.min(used >>> 1, 2)
          new BSV(new Array[Int](len), new Array[Count](len), 0, numTopics)
        }
        results(i) = (vid, counter)
        i += numThreads
      }
    })
    Await.ready(all0, 1.hour)

    val all = Future.traverse(lcSrcIds.indices.by(3).iterator)(lsi => Future {
      val si = lcSrcIds(lsi)
      val startPos = lcSrcIds(lsi + 1)
      val endPos = lcSrcIds(lsi + 2)
      val termTopics = results(si)._2
      var pos = startPos
      while (pos < endPos) {
        var ind = lcDstIds(pos)
        if (ind >= 0) {
          val di = ind
          val docTopics = results(di)._2
          val topic = data(pos)
          termTopics(topic) += 1
          docTopics.synchronized {
            docTopics(topic) += 1
          }
          pos += 1
        } else {
          val di = lcDstIds(pos + 1)
          val docTopics = results(di)._2
          while (ind < 0) {
            val topic = data(pos)
            termTopics(topic) += 1
            docTopics.synchronized {
              docTopics(topic) += 1
            }
            pos += 1
            ind += 1
          }
        }
      }
      termTopics match {
        case v: BDV[Count] =>
          val used = v.data.count(_ > 0)
          if (used < dscp) {
            results(si) = (l2g(si), toBSV(v, used))
          }
        case _ =>
      }
    })
    Await.ready(all, 1.hour)
    closePartExecutionContext()

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
    val thq = new ConcurrentLinkedQueue(0 until numThreads)
    val gens = Array.tabulate(numThreads)(thid => new XORShiftRandom(73 * numThreads + thid))
    @volatile var llhs = 0D
    @volatile var wllhs = 0D
    @volatile var dllhs = 0D
    val abDenseSum = sum_abDense(alphak_denoms, beta)

    implicit val es = initPartExecutionContext()
    val all = Future.traverse(lcSrcIds.indices.by(3).iterator)(lsi => Future {
      val thid = thq.poll()
      val si = lcSrcIds(lsi)
      val startPos = lcSrcIds(lsi + 1)
      val endPos = lcSrcIds(lsi + 2)
      val gen = gens(thid)
      val termTopics = vattrs(si)
      val waSparseSum = sum_waSparse(alphak_denoms, termTopics)
      val sum12 = abDenseSum + waSparseSum
      var llhs_th = 0D
      var wllhs_th = 0D
      var dllhs_th = 0D
      val common = isCommon(gen, startPos, endPos, lcDstIds, vattrs)
      var pos = startPos
      if (common) {
        val termBeta_denoms = calc_termBeta_denoms(denoms, beta_denoms, termTopics)
        while (pos < endPos) {
          var ind = lcDstIds(pos)
          if (ind >= 0) {
            val di = ind
            val docTopics = vattrs(di).asInstanceOf[Ndk]
            var doc_denom = doc_denoms(di)
            if (doc_denom == 0.0) {
              doc_denom = 1.0 / (sum(docTopics) + alphaSum)
              doc_denoms(di) = doc_denom
            }
            val dwbSparseSum = sum_dwbSparse_wOpt(termBeta_denoms, docTopics)
            llhs_th += Math.log((sum12 + dwbSparseSum) * doc_denom)
            val topic = data(pos)
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
            val dwbSparseSum = sum_dwbSparse_wOpt(termBeta_denoms, docTopics)
            llhs_th += Math.log((sum12 + dwbSparseSum) * doc_denom) * -ind
            while (ind < 0) {
              val topic = data(pos)
              wllhs_th += Math.log(termBeta_denoms(topic))
              dllhs_th += Math.log((docTopics(topic) + alphaks(topic)) * doc_denom)
              pos += 1
              ind += 1
            }
          }
        }
      } else {
        val denseTermTopics = toBDV(termTopics)
        while (pos < endPos) {
          var ind = lcDstIds(pos)
          if (ind >= 0) {
            val di = lcDstIds(pos)
            val docTopics = vattrs(di).asInstanceOf[Ndk]
            var doc_denom = doc_denoms(di)
            if (doc_denom == 0.0) {
              doc_denom = 1.0 / (sum(docTopics) + alphaSum)
              doc_denoms(di) = doc_denom
            }
            val dwbSparseSum = sum_dwbSparse(denoms, denseTermTopics, docTopics, beta)
            llhs_th += Math.log((sum12 + dwbSparseSum) * doc_denom)
            val topic = data(pos)
            wllhs_th += Math.log((denseTermTopics(topic) + beta) * denoms(topic))
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
            val dwbSparseSum = sum_dwbSparse(denoms, denseTermTopics, docTopics, beta)
            llhs_th += Math.log((sum12 + dwbSparseSum) * doc_denom) * -ind
            while (ind < 0) {
              val topic = data(pos)
              wllhs_th += Math.log((denseTermTopics(topic) + beta) * denoms(topic))
              dllhs_th += Math.log((docTopics(topic) + alphaks(topic)) * doc_denom)
              pos += 1
              ind += 1
            }
          }
        }
      }
      llhs += llhs_th
      wllhs += wllhs_th
      dllhs += dllhs_th
      thq.add(thid)
    })
    Await.ready(all, 2.hour)
    closePartExecutionContext()

    (llhs, wllhs, dllhs)
  }
}
