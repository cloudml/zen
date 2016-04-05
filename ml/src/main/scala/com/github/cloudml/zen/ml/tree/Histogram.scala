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

package com.github.cloudml.zen.ml.tree

class Histogram(val numBins: Int) {
  private val _counts = new Array[Double](numBins)
  private val _scores = new Array[Double](numBins)
  private val _squares = new Array[Double](numBins)
  private val _scoreWeights = new Array[Double](numBins)

  @inline def counts: Array[Double] = _counts

  @inline def scores: Array[Double] = _scores

  @inline def squares: Array[Double] = _squares

  @inline def scoreWeights: Array[Double] = _scoreWeights

  def weightedUpdate(bin: Int, score: Double, scoreWeight: Double, weight: Double = 1.0): Unit = {
    _counts(bin) += weight
    _scores(bin) += score * weight
    _squares(bin) += score * score * weight
    _scoreWeights(bin) += scoreWeight
  }

  def update(bin: Int, score: Double, scoreWeight: Double): Unit = {
    _counts(bin) += 1
    _scores(bin) += score
    _squares(bin) += score * score
    _scoreWeights(bin) += scoreWeight
  }

  def cumulateLeft(): Histogram = {
    var bin = 1
    while (bin < numBins) {
      _counts(bin) += _counts(bin-1)
      _scores(bin) += _scores(bin-1)
      _squares(bin) += _squares(bin-1)
      _scoreWeights(bin) += _scoreWeights(bin-1)
      bin += 1
    }
    this
  }

  def cumulate(info: NodeInfoStats): Histogram = {
    // cumulate from right to left
    var bin = numBins-2
    while (bin >0) {
      val binRight = bin + 1
      _counts(bin) += _counts(binRight)
      _scores(bin) += _scores(binRight)
      _squares(bin) += _squares(binRight)
      _scoreWeights(bin) += _scoreWeights(binRight)
      bin -= 1
    }

    // fill in Entry(0) with node sum information
    _counts(0)=info.sumCount
    _scores(0)=info.sumScores
    _squares(0)=info.sumSquares
    _scoreWeights(0)=info.sumScoreWeights

    this
  }
}

class NodeInfoStats(var sumCount: Int,
  var sumScores: Double,
  var sumSquares: Double,
  var sumScoreWeights: Double)extends Serializable {

  override def toString: String = s"NodeInfoStats($sumCount, $sumScores, $sumSquares, $sumScoreWeights)"

  def canEqual(other: Any): Boolean = other.isInstanceOf[NodeInfoStats]

  override def equals(other: Any): Boolean = other match {
    case that: NodeInfoStats =>
      (that canEqual this) &&
        sumCount == that.sumCount &&
        sumScores == that.sumScores &&
        sumSquares == that.sumSquares &&
        sumScoreWeights == that.sumScoreWeights
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(sumCount, sumScores, sumSquares, sumScoreWeights)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }
}
