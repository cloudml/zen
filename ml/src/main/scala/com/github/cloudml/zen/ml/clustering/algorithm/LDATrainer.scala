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

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, sum}
import com.github.cloudml.zen.ml.clustering.LDADefines._
import com.github.cloudml.zen.ml.sampler._
import com.github.cloudml.zen.ml.util.BVCompressor
import org.apache.spark.graphx2.impl.{ShippableVertexPartition => VertPartition}

import scala.concurrent._
import scala.concurrent.duration._


abstract class LDATrainer(numTopics: Int, numThreads: Int)
  extends LDAAlgorithm(numTopics, numThreads) {
  def aggregateCounters(vp: VertPartition[TC], cntsIter: Iterator[NvkPair]): VertPartition[TC] = {
    val totalSize = vp.capacity
    val index = vp.index
    val mask = vp.mask
    val values = vp.values
    val results = new Array[Nvk](totalSize)
    val marks = new AtomicIntegerArray(totalSize)
    implicit val es = ExecutionContext.fromExecutorService(Executors.newFixedThreadPool(numThreads))
    val all = cntsIter.grouped(numThreads * 5).map(batch => Future {
      batch.foreach(Function.tupled((vid, counter) => {
        val i = index.getPos(vid)
        if (marks.getAndDecrement(i) == 0) {
          results(i) = counter
        } else {
          while (marks.getAndSet(i, -1) <= 0) {}
          val agg = results(i)
          results(i) = if (isTermId(vid)) agg match {
            case u: BDV[Count] => counter match {
              case v: BDV[Count] => u :+= v
              case v: BSV[Count] => u :+= v
            }
            case u: BSV[Count] => counter match {
              case v: BDV[Count] => v :+= u
              case v: BSV[Count] =>
                u :+= v
                if (u.activeSize >= dscp) toBDV(u) else u
            }
          } else {
            agg.asInstanceOf[Ndk] :+= counter.asInstanceOf[Ndk]
          }
        }
        marks.set(i, Int.MaxValue)
      }))
    })
    Await.ready(Future.sequence(all), 1.hour)

    // compress counters
    val sizePerthrd = {
      val npt = totalSize / numThreads
      if (npt * numThreads == totalSize) npt else npt + 1
    }
    val all2 = Range(0, numThreads).map(thid => Future {
      val comp = new BVCompressor(numTopics)
      val startPos = sizePerthrd * thid
      val endPos = math.min(sizePerthrd * (thid + 1), totalSize)
      var pos = mask.nextSetBit(startPos)
      while (pos < endPos && pos >= 0) {
        values(pos) = comp.BV2CV(results(pos))
        pos = mask.nextSetBit(pos + 1)
      }
    })
    Await.ready(Future.sequence(all2), 1.hour)

    es.shutdown()
    vp.withValues(values)
  }

  def resetDist_abDense(ab: DiscreteSampler[Double],
    alphak_denoms: BDV[Double],
    beta: Double): DiscreteSampler[Double] = {
    val probs = alphak_denoms.copy :*= beta
    ab.resetDist(probs.data, null, probs.length)
  }

  @inline def sum_abDense(alphak_denoms: BDV[Double],
    beta: Double): Double = {
    sum(alphak_denoms.copy :*= beta)
  }

  def calc_denoms(topicCounters: BDV[Count],
    betaSum: Double): BDV[Double] = {
    val k = topicCounters.length
    val bdv = BDV.zeros[Double](k)
    var i = 0
    while (i < k) {
      bdv(i) = 1.0 / (topicCounters(i) + betaSum)
      i += 1
    }
    bdv
  }

  @inline def calc_alphak_denoms(denoms: BDV[Double],
    alphaAS: Double,
    betaSum: Double,
    alphaRatio: Double): BDV[Double] = {
    (denoms.copy :*= ((alphaAS - betaSum) * alphaRatio)) :+= alphaRatio
  }
}

object LDATrainer {
  def initAlgorithm(algoStr: String, numTopics: Int, numThreads: Int): LDATrainer = {
    algoStr.toLowerCase match {
      case "zenlda" =>
        println("using ZenLDA sampling algorithm.")
        new ZenLDA(numTopics, numThreads)
      case "lightlda" =>
        println("using LightLDA sampling algorithm.")
        new LightLDA(numTopics, numThreads)
      case "f+lda" =>
        println("using F+LDA sampling algorithm.")
        new FPlusLDA(numTopics, numThreads)
      case "sparselda" =>
        println("using SparseLDA sampling algorithm")
        new SparseLDA(numTopics, numThreads)
      case _ =>
        throw new NoSuchMethodException("No this algorithm or not implemented.")
    }
  }
}
