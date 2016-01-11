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

import java.lang.ref.SoftReference
import java.util.Random
import java.util.concurrent.atomic.AtomicIntegerArray
import java.util.concurrent.{ConcurrentLinkedQueue, Executors}

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, sum}
import com.github.cloudml.zen.ml.clustering.LDADefines._
import com.github.cloudml.zen.ml.clustering.LDAPerplexity
import com.github.cloudml.zen.ml.sampler._
import com.github.cloudml.zen.ml.util.XORShiftRandom
import me.lemire.integercompression.IntCompressor
import me.lemire.integercompression.differential.IntegratedIntCompressor
import org.apache.spark.graphx2._
import org.apache.spark.graphx2.impl.{ShippableVertexPartition => VertPartition, EdgePartition, GraphImpl}

import scala.collection.JavaConversions._
import scala.concurrent._
import scala.concurrent.duration._


abstract class LDATrainer(numTopics: Int, numThreads: Int)
  extends LDAAlgorithm(numTopics, numThreads) {

  override def aggregateCounters(svp: VertPartition[TC],
    cntsIter: Iterator[NvkPair]): VertPartition[TC] = {
    val totalSize = svp.capacity
    val results = new Array[BV[Count]](totalSize)
    val index = svp.index
    val marks = new AtomicIntegerArray(results.length)
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
          } else agg match {
            case u: BSV[Count] => counter match {
              case v: BSV[Count] => u :+= v
            }
          }
        }
        marks.set(i, Int.MaxValue)
      }))
    })
    Await.ready(Future.sequence(all), 1.hour)

    val mask = svp.mask
    val values = svp.values
    val sizePerthrd = {
      val npt = totalSize / numThreads
      if (npt * numThreads == totalSize) npt else npt + 1
    }
    val all2 = Range(0, numThreads).map(thid => Future {
      val nic = new IntCompressor
      val iic = new IntegratedIntCompressor
      val startPos = sizePerthrd * thid
      val endPos = math.min(sizePerthrd * (thid + 1), totalSize)
      var pos = mask.nextSetBit(startPos)
      while (pos < endPos && pos >= 0) results(pos) match {
        case v: BDV[Count] =>
          val cdata = nic.compress(v.data)
          results(pos) = (cdata,)
        case v: BSV[Count] =>
          val cdata = nic.compress(v.data)
          val cindex = iic.compress(v.index)
          results(pos) = (cdata, cindex)
          pos = mask.nextSetBit(pos + 1)
      }
    })

    es.shutdown()
    svp.withValues(results)
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
  def initAlgorithm(algoStr: String): LDATrainer = {
    algoStr.toLowerCase match {
      case "zenlda" =>
        println("using ZenLDA sampling algorithm.")
        new ZenLDA
      case "lightlda" =>
        println("using LightLDA sampling algorithm.")
        new LightLDA
      case "sparselda" =>
        println("using SparseLDA sampling algorithm")
        new SparseLDA
      case _ =>
        throw new NoSuchMethodException("No this algorithm or not implemented.")
    }
  }
}
