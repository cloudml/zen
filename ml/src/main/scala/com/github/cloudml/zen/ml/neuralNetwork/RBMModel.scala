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

package com.github.cloudml.zen.ml.neuralNetwork

import java.util.Random

import breeze.linalg.{Axis => BrzAxis, DenseMatrix => BDM, DenseVector => BDV, sum => brzSum}
import com.github.cloudml.zen.ml.util._
import org.apache.commons.math3.random.JDKRandomGenerator
import org.apache.spark.mllib.util.Saveable
import org.apache.spark.SparkContext
import org.apache.spark.annotation.Experimental

@Experimental
class RBMModel(
  val weight: BDM[Double],
  val visibleBias: BDV[Double],
  val hiddenBias: BDV[Double],
  val dropoutRate: Double,
  val visibleLayerType: String = "ReLu",
  val hiddenLayerType: String = "ReLu") extends Saveable with Logging with Serializable {

  def this(
    numIn: Int,
    numOut: Int,
    dropout: Double) {
    this(NNUtil.initUniformDistWeight(numIn, numOut, math.sqrt(6D / (numIn + numOut))),
      NNUtil.initializeBias(numIn),
      NNUtil.initializeBias(numOut),
      dropout)
  }

  def this(
    numIn: Int,
    numOut: Int) {
    this(numIn, numOut, 0.5D)
  }

  require(dropoutRate >= 0 && dropoutRate < 1)
  @transient protected lazy val rand: Random = new JDKRandomGenerator()
  @transient protected[ml] lazy val visibleLayer: Layer = {
    Layer.initializeLayer(weight.t, visibleBias, visibleLayerType)
  }

  @transient protected[ml] lazy val hiddenLayer: Layer = {
    Layer.initializeLayer(weight, hiddenBias, visibleLayerType)
  }

  setSeed(Utils.random.nextInt())

  def setSeed(seed: Long): Unit = {
    rand.setSeed(seed)
    visibleLayer.setSeed(rand.nextInt())
    hiddenLayer.setSeed(rand.nextInt())
  }

  def cdK: Int = 1

  def numOut: Int = weight.rows

  def numIn: Int = weight.cols

  def forward(visible: BDM[Double]): BDM[Double] = {
    val hidden = activateHidden(visible)
    if (dropoutRate > 0) {
      hidden :*= (1 - dropoutRate)
    }
    hidden
  }

  protected def activateHidden(visible: BDM[Double]): BDM[Double] = {
    require(visible.rows == weight.cols)
    hiddenLayer.forward(visible)
  }

  protected def sampleHidden(hiddenMean: BDM[Double]): BDM[Double] = {
    hiddenLayer.sample(hiddenMean)
  }

  protected def sampleVisible(visibleMean: BDM[Double]): BDM[Double] = {
    visibleLayer.sample(visibleMean)
  }

  protected def activateVisible(hidden: BDM[Double]): BDM[Double] = {
    require(hidden.rows == weight.rows)
    visibleLayer.forward(hidden)
  }

  protected def dropOutMask(cols: Int): BDM[Double] = {
    val mask = BDM.zeros[Double](numOut, cols)
    for (i <- 0 until numOut) {
      for (j <- 0 until cols) {
        mask(i, j) = if (rand.nextDouble() > dropoutRate) 1D else 0D
      }
    }
    mask
  }

  private[ml] def learn(input: BDM[Double]): (BDM[Double], BDV[Double], BDV[Double], Double, Double) = {
    val batchSize = input.cols
    val mask: BDM[Double] = if (dropoutRate > 0) {
      dropOutMask(batchSize)
    } else {
      null
    }

    val h1Mean = activateHidden(input)
    val h1Sample = sampleHidden(h1Mean)

    var vKMean: BDM[Double] = null
    var vKSample: BDM[Double] = null
    var hKMean: BDM[Double] = null
    var hKSample: BDM[Double] = h1Sample
    if (dropoutRate > 0) {
      hKSample :*= mask
    }

    for (i <- 0 until cdK) {
      vKMean = activateVisible(hKSample)
      hKMean = activateHidden(vKMean)
      hKSample = sampleHidden(hKMean)
      if (dropoutRate > 0) {
        hKSample :*= mask
      }
    }

    val gradWeight: BDM[Double] = hKMean * vKMean.t - h1Mean * input.t
    val gradVisibleBias = brzSum(vKMean - input, BrzAxis._1)
    val gradHiddenBias = brzSum(hKMean - h1Mean, BrzAxis._1)

    val mse = NNUtil.meanSquaredError(input, vKMean)
    (gradWeight, gradVisibleBias, gradHiddenBias, mse, batchSize.toDouble)
  }

  protected def formatVersion: String = RBM.SaveLoadV1_0.formatVersionV1_0

  override def save(sc: SparkContext, path: String): Unit = {
    RBM.SaveLoadV1_0.save(sc, path, this)
  }

  override def equals(other: Any): Boolean = other match {
    case l: RBMModel =>
      visibleLayerType == l.visibleLayerType && hiddenLayerType == l.hiddenLayerType &&
        weight == l.weight && visibleBias == l.visibleBias && hiddenBias == l.hiddenBias &&
        dropoutRate == l.dropoutRate
    case _ =>
      false
  }

  override def hashCode: Int = {
    import com.google.common.base.Objects
    31 * Objects.hashCode(visibleLayerType, hiddenLayerType, weight,
      visibleBias, hiddenBias) + dropoutRate.hashCode()
  }
}
