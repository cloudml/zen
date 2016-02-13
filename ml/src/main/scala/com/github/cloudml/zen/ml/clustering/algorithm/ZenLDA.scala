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
import com.github.cloudml.zen.ml.util.Concurrent._
import com.github.cloudml.zen.ml.util.XORShiftRandom
import org.apache.spark.graphx2.impl.EdgePartition

import scala.collection.JavaConversions._
import scala.concurrent.Future


class ZenLDA(numTopics: Int, numThreads: Int)
  extends LDATrainerByWord(numTopics, numThreads) {
  override def initEdgePartition(ep: EdgePartition[TA, _]): EdgePartition[TA, Int] = {
    val totalSize = ep.size
    val srcSize = ep.indexSize
    val lcSrcIds = ep.localSrcIds
    val lcDstIds = ep.localDstIds
    val zeros = new Array[Int](ep.vertexAttrs.length)
    val srcInfos = new Array[(Int, Int, Int)](srcSize)

    implicit val es = initExecutionContext(numThreads)
    val all = Future.traverse(ep.index.iterator.zipWithIndex) { case ((_, startPos), ii) => withFuture {
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
    withAwaitReadyAndClose(all)

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

    val global = new AliasTable[Double]
    val thq = new ConcurrentLinkedQueue(1 to numThreads)
    val gens = new Array[XORShiftRandom](numThreads)
    val termDists = new Array[AliasTable[Double]](numThreads)
    val cdfDists = new Array[CumulativeDist[Double]](numThreads)
    resetDist_abDense(global, alphak_denoms, beta)

    implicit val es = initExecutionContext(numThreads)
    val all = Future.traverse(lcSrcIds.indices.by(3).iterator) { lsi => withFuture {
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
      } finally {
        thq.add(thid + 1)
      }
    }}
    withAwaitReadyAndClose(all)

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
    ab: AliasTable[Double],
    wa: AliasTable[Double],
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
    ab: AliasTable[Double],
    wa: AliasTable[Double],
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

    val all = Future.traverse(lcSrcIds.indices.by(3).iterator) { lsi => withFuture {
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
    }}
    withAwaitReadyAndClose(all)

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
    val docNorms = new Array[Double](vertSize)
    val thq = new ConcurrentLinkedQueue(1 to numThreads)
    val gens = Array.tabulate(numThreads)(thid => new XORShiftRandom(73 * numThreads + thid))
    @volatile var llhs = 0.0
    @volatile var wllhs = 0.0
    @volatile var dllhs = 0.0
    val abDenseSum = sum_abDense(alphak_denoms, beta)

    implicit val es = initExecutionContext(numThreads)
    val all = Future.traverse(lcSrcIds.indices.by(3).iterator) { lsi => withFuture {
      val thid = thq.poll() - 1
      try {
        val si = lcSrcIds(lsi)
        val startPos = lcSrcIds(lsi + 1)
        val endPos = lcSrcIds(lsi + 2)
        val gen = gens(thid)
        val termTopics = vattrs(si)
        val waSparseSum = sum_waSparse(alphak_denoms, termTopics)
        val sum12 = abDenseSum + waSparseSum
        var llhs_th = 0.0
        var wllhs_th = 0.0
        var dllhs_th = 0.0
        val common = isCommon(gen, startPos, endPos, lcDstIds, vattrs)
        var pos = startPos
        if (common) {
          val termBeta_denoms = calc_termBeta_denoms(denoms, beta_denoms, termTopics)
          while (pos < endPos) {
            var ind = lcDstIds(pos)
            if (ind >= 0) {
              val di = ind
              val docTopics = vattrs(di).asInstanceOf[Ndk]
              var docNorm = docNorms(di)
              if (docNorm == 0.0) {
                docNorm = 1.0 / (sum(docTopics) + alphaSum)
                docNorms(di) = docNorm
              }
              val dwbSparseSum = sum_dwbSparse_wOpt(termBeta_denoms, docTopics)
              llhs_th += Math.log((sum12 + dwbSparseSum) * docNorm)
              val topic = data(pos)
              wllhs_th += Math.log(termBeta_denoms(topic))
              dllhs_th += Math.log((docTopics(topic) + alphaks(topic)) * docNorm)
              pos += 1
            } else {
              val di = lcDstIds(pos + 1)
              val docTopics = vattrs(di).asInstanceOf[Ndk]
              var docNorm = docNorms(di)
              if (docNorm == 0.0) {
                docNorm = 1.0 / (sum(docTopics) + alphaSum)
                docNorms(di) = docNorm
              }
              val dwbSparseSum = sum_dwbSparse_wOpt(termBeta_denoms, docTopics)
              llhs_th += Math.log((sum12 + dwbSparseSum) * docNorm) * -ind
              while (ind < 0) {
                val topic = data(pos)
                wllhs_th += Math.log(termBeta_denoms(topic))
                dllhs_th += Math.log((docTopics(topic) + alphaks(topic)) * docNorm)
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
              var docNorm = docNorms(di)
              if (docNorm == 0.0) {
                docNorm = 1.0 / (sum(docTopics) + alphaSum)
                docNorms(di) = docNorm
              }
              val dwbSparseSum = sum_dwbSparse(denoms, denseTermTopics, docTopics, beta)
              llhs_th += Math.log((sum12 + dwbSparseSum) * docNorm)
              val topic = data(pos)
              wllhs_th += Math.log((denseTermTopics(topic) + beta) * denoms(topic))
              dllhs_th += Math.log((docTopics(topic) + alphaks(topic)) * docNorm)
              pos += 1
            } else {
              val di = lcDstIds(pos + 1)
              val docTopics = vattrs(di).asInstanceOf[Ndk]
              var docNorm = docNorms(di)
              if (docNorm == 0.0) {
                docNorm = 1.0 / (sum(docTopics) + alphaSum)
                docNorms(di) = docNorm
              }
              val dwbSparseSum = sum_dwbSparse(denoms, denseTermTopics, docTopics, beta)
              llhs_th += Math.log((sum12 + dwbSparseSum) * docNorm) * -ind
              while (ind < 0) {
                val topic = data(pos)
                wllhs_th += Math.log((denseTermTopics(topic) + beta) * denoms(topic))
                dllhs_th += Math.log((docTopics(topic) + alphaks(topic)) * docNorm)
                pos += 1
                ind += 1
              }
            }
          }
        }
        llhs += llhs_th
        wllhs += wllhs_th
        dllhs += dllhs_th
      } finally {
        thq.add(thid + 1)
      }
    }}
    withAwaitReadyAndClose(all)

    (llhs, wllhs, dllhs)
  }

  def resetDist_abDense(ab: AliasTable[Double],
    alphak_denoms: BDV[Double],
    beta: Double): AliasTable[Double] = {
    val probs = alphak_denoms.copy :*= beta
    ab.resetDist(probs.data, null, probs.length)
  }

  def resetDist_waSparse(wa: AliasTable[Double],
    alphak_denoms: BDV[Double],
    termTopics: Nwk): AliasTable[Double] = termTopics match {
    case v: BDV[Count] =>
      val probs = new Array[Double](numTopics)
      val space = new Array[Int](numTopics)
      var psize = 0
      var i = 0
      while (i < numTopics) {
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

  def resetDist_dwbSparse(dwb: CumulativeDist[Double],
    denoms: BDV[Double],
    denseTermTopics: BDV[Count],
    docTopics: Ndk,
    beta: Double): CumulativeDist[Double] = {
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
      sum += (denseTermTopics(topic) + beta) * data(i) * denoms(topic)
      cdf(i) = sum
      i += 1
    }
    dwb._space = index
    dwb
  }

  def resetDist_dwbSparse_wAdjust(dwb: CumulativeDist[Double],
    denoms: BDV[Double],
    denseTermTopics: BDV[Count],
    docTopics: Ndk,
    curTopic: Int,
    beta: Double): CumulativeDist[Double] = {
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
      val docCnt = data(i)
      val termBeta = denseTermTopics(topic) + beta
      val numer = if (topic == curTopic) {
        (termBeta - 1.0) * (docCnt - 1)
      } else {
        termBeta * docCnt
      }
      sum += numer * denoms(topic)
      cdf(i) = sum
      i += 1
    }
    dwb._space = index
    dwb
  }

  def resetDist_dwbSparse_wOpt(dwb: CumulativeDist[Double],
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

  def resetDist_dwbSparse_wOptAdjust(dwb: CumulativeDist[Double],
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
      val docCnt = data(i)
      val termBeta_denom = termBeta_denoms(topic)
      val prob = if (topic == curTopic) {
        (termBeta_denom - denoms(topic)) * (docCnt - 1)
      } else {
        termBeta_denom * docCnt
      }
      sum += prob
      cdf(i) = sum
      i += 1
    }
    dwb._space = index
    dwb
  }

  def calc_termBeta_denoms(denoms: BDV[Double],
    beta_denoms: BDV[Double],
    termTopics: Nwk): BDV[Double] = {
    val bdv = beta_denoms.copy
    termTopics match {
      case v: BDV[Count] =>
        var i = 0
        while (i < numTopics) {
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
