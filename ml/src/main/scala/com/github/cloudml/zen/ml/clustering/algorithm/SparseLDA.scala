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
    val thq = new ConcurrentLinkedQueue(1 to numThreads)
    val gens = new Array[XORShiftRandom](numThreads)
    val docDists = new Array[FlatDist[Double]](numThreads)
    val mainDists = new Array[FlatDist[Double]](numThreads)
    val compSamps = new Array[CompositeSampler](numThreads)
    resetDist_abDense(global, topicCounters, alphaAS, beta, betaSum, alphaRatio)

    implicit val es = initExecutionContext(numThreads)
    val all = Future.traverse(ep.index.iterator) { case (_, startPos) => withFuture {
      val thid = thq.poll() - 1
      try {
        var gen = gens(thid)
        if (gen == null) {
          gen = new XORShiftRandom(((seed + sampIter) * numPartitions + pid) * numThreads + thid)
          gens(thid) = gen
          docDists(thid) = new FlatDist[Double](isSparse = true).reset(numTopics)
          mainDists(thid) = new FlatDist[Double](isSparse = true).reset(numTopics)
          compSamps(thid) = new CompositeSampler
        }
        val docDist = docDists(thid)
        val mainDist = mainDists(thid)
        val compSamp = compSamps(thid)

        val si = lcSrcIds(startPos)
        val docTopics = vattrs(si).asInstanceOf[Ndk]
        useds(si) = docTopics.activeSize
        resetDist_dbSparse(docDist, topicCounters, docTopics, beta, betaSum)
        val updateCurry = updateTopicAssign(global, docDist, topicCounters, alphaAS, beta, betaSum, alphaRatio) _
        var pos = startPos
        while (pos < totalSize && lcSrcIds(pos) == si) {
          val di = lcDstIds(pos)
          val termTopics = vattrs(di)
          useds(di) = termTopics.activeSize
          val tokenUpdate = updateCurry(termTopics, docTopics)
          val topic = data(pos)
          tokenUpdate(topic, -1)
          resetDist_wdaSparse(mainDist, topicCounters, termTopics, docTopics, alphaAS, betaSum, alphaRatio, topic)
          compSamp.resetComponents(mainDist, docDist, global)
          val newTopic = compSamp.sampleRandom(gen)
          tokenUpdate(newTopic, 1)
          data(pos) = newTopic
          pos += 1
        }
      } finally {
        thq.add(thid + 1)
      }
    }}
    withAwaitReadyAndClose(all)

    ep.withVertexAttributes(useds)
  }

  def updateTopicAssign(ab: FlatDist[Double],
    db: FlatDist[Double],
    topicCounters: BDV[Count],
    alphaAS: Double,
    beta: Double,
    betaSum: Double,
    alphaRatio: Double)
    (termTopics: Nwk, docTopics: Ndk)
    (topic: Int, delta: Int): Unit = {
    topicCounters(topic) += delta
    docTopics(topic) += delta
    termTopics.synchronized {
      termTopics(topic) += delta
    }
    val ns = topicCounters(topic)
    val nds = docTopics(topic)
    val betaDenom = beta / (ns + betaSum)
    val alphak = (ns + alphaAS) * alphaRatio
    ab(topic) = alphak * betaDenom
    db(topic) = nds * betaDenom
  }

  def resetDist_abDense(ab: FlatDist[Double],
    topicCounters: BDV[Count],
    alphaAS: Double,
    beta: Double,
    betaSum: Double,
    alphaRatio: Double): FlatDist[Double] = {
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
    db.resetDist(probs, index.clone(), used)
  }

  def resetDist_wdaSparse(wda: FlatDist[Double],
    topicCounters: BDV[Count],
    termTopics: Nwk,
    docTopics: Ndk,
    alphaAS: Double,
    betaSum: Double,
    alphaRatio: Double,
    curTopic: Int): FlatDist[Double] = {
    val tmpTermTopics = termTopics.synchronized(termTopics.copy)
    tmpTermTopics match {
      case v: BDV[Count] =>
        val probs = new Array[Double](numTopics)
        val space = new Array[Int](numTopics)
        var psize = 0
        var i = 0
        while (i < numTopics) {
          val cnt = v(i)
          if (cnt > 0) {
            val nk = topicCounters(i)
            val alphak = (nk + alphaAS) * alphaRatio
            probs(psize) = (docTopics(i) + alphak) * cnt / (nk + betaSum)
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
          val nk = topicCounters(topic)
          val alphak = (nk + alphaAS) * alphaRatio
          probs(i) = (docTopics(topic) + alphak) * data(i) / (nk + betaSum)
          i += 1
        }
        wda.resetDist(probs, index, used)
    }
  }
}
