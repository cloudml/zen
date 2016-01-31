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

package com.github.cloudml.zen.ml.clustering

trait LDAMetrics {
  def getTotal: Double
  def getWord: Double
  def getDoc: Double
  def output(writer: String => Unit): Unit
}

class LDAPerplexity(val pplx: Double, val wpplx: Double, val dpplx: Double) extends LDAMetrics {
  override def getTotal: Double = pplx

  override def getWord: Double = wpplx

  override def getDoc: Double = dpplx

  override def output(writer: String => Unit): Unit = {
    val o = s"perplexity=$getTotal, word pplx=$getWord, doc pplx=$getDoc"
    writer(o)
  }
}

class LDALogLikelihood(val wllh: Double, val dllh: Double) extends LDAMetrics {
  override def getTotal: Double = wllh + dllh

  override def getWord: Double = wllh

  override def getDoc: Double = dllh

  override def output(writer: String => Unit): Unit = {
    val o = s"total llh=$getTotal, word llh=$getWord, doc llh=$getDoc"
    writer(o)
  }
}

object LDAMetrics {
  def apply(evalMetric: String, lda: LDA): LDAMetrics = {
    val verts = lda.verts
    val topicCounters = lda.topicCounters
    val numTokens = lda.numTokens
    val numTerms = lda.numTerms
    val alpha = lda.alpha
    val alphaAS = lda.alphaAS
    val beta = lda.beta
    evalMetric match {
      case "pplx" =>
        lda.algo.calcPerplexity(lda.edges, verts, topicCounters, numTokens, numTerms, alpha, alphaAS, beta)
      case "llh" =>
        lda.algo.calcLogLikelihood(verts, topicCounters, numTokens, lda.numDocs, numTerms, alpha, alphaAS, beta)
    }
  }
}
