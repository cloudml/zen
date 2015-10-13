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

trait LDAMetrics

class LDAPerplexity(val pplx: Double, val wpplx: Double, val dpplx: Double) extends LDAMetrics {

  @inline def getPerplexity: Double = pplx

  @inline def getWordPerplexity: Double = wpplx

  @inline def getDocPerplexity: Double = dpplx

  def output(writer: String => Unit) = {
    val o = s"perplexity=$pplx, word pplx=$wpplx, doc pplx=$dpplx"
    writer(o)
  }
}

object LDAPerplexity {
  def apply(lda: LDA): LDAPerplexity = {
    val corpus = lda.corpus
    val topicCounters = lda.topicCounters
    val numTokens = lda.numTokens
    val numTopics = lda.numTopics
    val numTerms = lda.numTerms
    val alpha = lda.alpha
    val alphaAS = lda.alphaAS
    val beta = lda.beta
    lda.algo.calcPerplexity(corpus, topicCounters, numTokens, numTopics, numTerms,
      alpha, alphaAS, beta)
  }
}
