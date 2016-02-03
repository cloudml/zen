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

import java.util.concurrent.ConcurrentLinkedQueue

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV}
import com.github.cloudml.zen.ml.clustering.LDADefines._
import com.github.cloudml.zen.ml.sampler._
import com.github.cloudml.zen.ml.util.Concurrent._
import com.github.cloudml.zen.ml.util.XORShiftRandom
import org.apache.spark.graphx2.impl.EdgePartition

import scala.collection.JavaConversions._
import scala.concurrent.Future


class SparseLDA(numTopics: Int, numThreads: Int)
  extends LDATrainerByDoc(numTopics: Int, numThreads: Int) {
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

    val totalSize = ep.size
    val lcSrcIds = ep.localSrcIds
    val lcDstIds = ep.localDstIds
    val vattrs = ep.vertexAttrs
    val useds = new Array[Int](vattrs.length)
    val data = ep.data

    val global = new FlatDist[Double](isSparse=false)
    val thq = new ConcurrentLinkedQueue(0 until numThreads)
    val gens = new Array[XORShiftRandom](numThreads)
    val docDists = new Array[FlatDist[Double]](numThreads)
    val mainDists = new Array[FlatDist[Double]](numThreads)
    val compSamps = new Array[CompositeSampler](numThreads)
    resetDist_abDense(global, topicCounters, alphaAS, beta, betaSum, alphaRatio)

    implicit val es = initExecutionContext(numThreads)
    val all = Future.traverse(ep.index.iterator) { case (_, startPos) => withFuture {
      val thid = thq.poll()
      var gen = gens(thid)
      if (gen == null) {
        gen = new XORShiftRandom(((seed + sampIter) * numPartitions + pid) * numThreads + thid)
        gens(thid) = gen
        docDists(thid) = new FlatDist[Double](isSparse=true).reset(numTopics)
        mainDists(thid) = new FlatDist[Double](isSparse=true).reset(numTopics)
        compSamps(thid) = new CompositeSampler
      }
      val docDist = docDists(thid)
      val mainDist = mainDists(thid)
      val compSamp = compSamps(thid)

      val si = lcSrcIds(startPos)
      val docTopics = vattrs(si).asInstanceOf[BSV[Count]]
      useds(si) = docTopics.activeSize
      resetDist_dbSparse(docDist, topicCounters, docTopics, beta, betaSum)
      var pos = startPos
      while (pos < totalSize && lcSrcIds(pos) == si) {
        val di = lcDstIds(pos)
        val termTopics = vattrs(di)
        useds(di) = termTopics.activeSize
        val topic = data(pos)
        topicCounters(topic) -= 1
        docTopics(topic) -= 1
        termTopics.synchronized {
          termTopics(topic) -= 1
        }
        global.update(topic, (topicCounters(topic) + alphaAS) * alphaRatio * beta / (topicCounters(topic) + betaSum))
        docDist.update(topic, docTopics(topic) * beta / (topicCounters(topic) + betaSum))
        resetDist_wdaSparse_wAdjust(mainDist, topicCounters, termTopics, docTopics, alphaAS, betaSum, alphaRatio, topic)
        compSamp.resetComponents(mainDist, docDist, global)
        val newTopic = compSamp.sampleRandom(gen)
        data(pos) = newTopic
        topicCounters(newTopic) += 1
        docTopics(newTopic) += 1
        termTopics.synchronized {
          termTopics(newTopic) += 1
        }
        global.update(newTopic,
          (topicCounters(newTopic) + alphaAS) * alphaRatio * beta / (topicCounters(newTopic) + betaSum))
        docDist.update(newTopic, docTopics(newTopic) * beta / (topicCounters(newTopic) + betaSum))
        pos += 1
      }
      thq.add(thid)
    }}
    withAwaitReadyAndClose(all)

    ep.withVertexAttributes(useds)
  }

  def resetDist_abDense(ab: FlatDist[Double],
    topicCounters: BDV[Count],
    alphaAS: Double,
    beta: Double,
    betaSum: Double,
    alphaRatio: Double): DiscreteSampler[Double] = {
    val probs = new Array[Double](numTopics)
    var i = 0
    while (i < numTopics) {
      val nk = topicCounters(i)
      val alphak = (nk + alphaAS) * alphaRatio
      probs(i) = alphak * beta / (nk + betaSum)
      i += 1
    }
    ab.synchronized {
      ab.resetDist(probs, null, numTopics)
    }
  }

  def resetDist_dbSparse(db: FlatDist[Double],
    topicCounters: BDV[Count],
    docTopics: Ndk,
    beta: Double,
    betaSum: Double): FlatDist[Double] = {
    val used = docTopics.used
    val index = docTopics.index
    val data = docTopics.data
    val probs = new Array[Double](used)
    var i = 0
    while (i < used) {
      probs(i) = data(i) * beta / (topicCounters(index(i)) + betaSum)
      i += 1
    }
    db.resetDist(probs, index, used)
  }

  def resetDist_wdaSparse_wAdjust(wda: FlatDist[Double],
    topicCounters: BDV[Count],
    termTopics: Nwk,
    docTopics: Ndk,
    alphaAS: Double,
    betaSum: Double,
    alphaRatio: Double,
    curTopic: Int): FlatDist[Double] = termTopics match {
    case v: BDV[Count] =>
      val probs = new Array[Double](numTopics)
      val space = new Array[Int](numTopics)
      var psize = 0
      var i = 0
      while (i < numTopics) {
        val cnt = v(i)
        if (cnt > 0) {
          val adjust = if (i == curTopic) -1 else 0
          val nk = topicCounters(i)
          val alphak = (nk + alphaAS) * alphaRatio
          probs(psize) = (docTopics(i) + adjust + alphak) * (cnt + adjust) / (nk + adjust + betaSum)
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
        val topic = index(i)
        val adjust = if (topic == curTopic) -1 else 0
        val nk = topicCounters(topic)
        val alphak = (nk + alphaAS) * alphaRatio
        probs(i) = (docTopics(topic) + adjust + alphak) * (data(i) + adjust) / (nk + adjust + betaSum)
        i += 1
      }
      wda.resetDist(probs, index, used)
  }
}
