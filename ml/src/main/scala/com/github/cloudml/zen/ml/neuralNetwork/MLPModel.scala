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

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import breeze.numerics.signum
import com.github.cloudml.zen.ml.util.Logging
import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.util.Saveable
import org.apache.spark.SparkContext

@Experimental
class MLPModel(
  val innerLayers: Array[Layer],
  val dropout: Array[Double]) extends Saveable with Logging with Serializable {
  def this(topology: Array[Int],
    inputDropout: Double,
    hiddenDropout: Double) {
    this(MLP.initLayers(topology),
      MLP.initDropout(topology.length - 1, Array(hiddenDropout, inputDropout)))
  }

  def this(topology: Array[Int]) {
    this(MLP.initLayers(topology), MLP.initDropout(topology.length - 1, Array(0.2, 0.5)))
  }

  require(innerLayers.length > 0)
  require(dropout.forall(t => t >= 0 && t < 1))
  require(dropout.last == 0D)
  require(innerLayers.length == dropout.length)

  @transient protected lazy val rand: Random = new Random()

  def topology: Array[Int] = {
    val topology = new Array[Int](numLayer + 1)
    topology(0) = numInput
    for (i <- 1 to numLayer) {
      topology(i) = innerLayers(i - 1).numOut
    }
    topology
  }

  def numLayer: Int = innerLayers.length

  def numInput: Int = innerLayers.head.numIn

  def numOut: Int = innerLayers.last.numOut

  def predict(x: BDM[Double]): BDM[Double] = {
    var output = x
    for (layer <- 0 until numLayer) {
      output = innerLayers(layer).forward(output)
      val dropoutRate = dropout(layer)
      if (dropoutRate > 0D) {
        output :*= (1D - dropoutRate)
      }
    }
    output
  }

  protected[ml] def computeDelta(
    x: BDM[Double],
    label: BDM[Double]): (Array[BDM[Double]], Array[BDM[Double]]) = {
    val batchSize = x.cols
    val out = new Array[BDM[Double]](numLayer)
    val delta = new Array[BDM[Double]](numLayer)
    val dropOutMasks: Array[BDM[Double]] = dropOutMask(batchSize)

    for (layer <- 0 until numLayer) {
      val output = innerLayers(layer).forward(if (layer == 0) x else out(layer - 1))
      if (dropOutMasks(layer) != null) {
        assert(output.rows == dropOutMasks(layer).rows)
        output :*= dropOutMasks(layer)
      }
      out(layer) = output
    }

    for (layer <- (0 until numLayer).reverse) {
      val output = out(layer)
      val currentLayer = innerLayers(layer)
      delta(layer) = if (layer == numLayer - 1) {
        currentLayer.outputError(output, label)
      } else {
        val nextLayer = innerLayers(layer + 1)
        val nextDelta = delta(layer + 1)
        nextLayer.previousError(output, currentLayer, nextDelta)
      }
      if (dropOutMasks(layer) != null) {
        delta(layer) :*= dropOutMasks(layer)
      }
    }
    (out, delta)
  }

  protected[ml] def computeGradient(
    x: BDM[Double],
    label: BDM[Double],
    epsilon: Double = 0.0): (Array[(BDM[Double], BDV[Double])], Double, Double) = {
    var input = x
    var (out, delta) = computeDelta(x, label)

    // Improving Back-Propagation by Adding an Adversarial Gradient
    // URL: http://arxiv.org/abs/1510.04189
    if (epsilon > 0.0) {
      var sign: BDM[Double] = innerLayers.head.weight.t * delta.head
      sign = signum(sign)
      sign :*= epsilon
      sign :+= x
      val t = computeDelta(sign, label)
      out = t._1
      delta = t._2
      input = sign
    }

    val grads = computeGradientGivenDelta(input, out, delta)
    val cost = if (innerLayers.last.layerType == "SoftMax") {
      NNUtil.crossEntropy(out.last, label)
    } else {
      NNUtil.meanSquaredError(out.last, label)
    }
    (grads, cost, input.cols.toDouble)
  }

  protected[ml] def computeGradientGivenDelta(
    x: BDM[Double],
    out: Array[BDM[Double]],
    delta: Array[BDM[Double]]): Array[(BDM[Double], BDV[Double])] = {
    val grads = new Array[(BDM[Double], BDV[Double])](numLayer)
    for (i <- 0 until numLayer) {
      val input = if (i == 0) x else out(i - 1)
      grads(i) = innerLayers(i).backward(input, delta(i))
    }
    grads
  }

  protected[ml] def dropOutMask(cols: Int): Array[BDM[Double]] = {
    val masks = new Array[BDM[Double]](numLayer)
    for (layer <- 0 until numLayer) {
      val dropoutRate = dropout(layer)
      masks(layer) = if (dropoutRate > 0) {
        val rows = innerLayers(layer).numOut
        val mask = BDM.zeros[Double](rows, cols)
        for (i <- 0 until rows) {
          for (j <- 0 until cols) {
            mask(i, j) = if (rand.nextDouble() > dropoutRate) 1D else 0D
          }
        }
        mask
      } else {
        null
      }
    }
    masks
  }

  protected[ml] def assign(newNN: MLPModel): MLPModel = {
    innerLayers.zip(newNN.innerLayers).foreach { case (oldLayer, newLayer) =>
      oldLayer.weight := newLayer.weight
      oldLayer.bias := newLayer.bias
    }
    this
  }

  def setSeed(seed: Long): Unit = {
    rand.setSeed(seed)
  }

  protected def formatVersion: String = MLP.SaveLoadV1_0.formatVersionV1_0

  override def save(sc: SparkContext, path: String): Unit = {
    MLP.SaveLoadV1_0.save(sc, path, this)
  }

  override def equals(other: Any): Boolean = other match {
    case m: MLPModel =>
      innerLayers.sameElements(m.innerLayers) && dropout.sameElements(m.dropout)
    case _ =>
      false
  }

  override def hashCode: Int = {
    var result: Int = 1
    for (element <- innerLayers) {
      result = 31 * result + (if (element == null) 0 else element.hashCode)
    }
    for (element <- dropout) {
      result = 31 * result + element.hashCode
    }

    return result
  }
}
