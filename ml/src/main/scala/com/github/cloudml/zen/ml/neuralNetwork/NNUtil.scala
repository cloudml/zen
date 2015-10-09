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

import breeze.linalg.{Axis => BrzAxis, DenseMatrix => BDM, DenseVector => BDV, Matrix => BM, max => brzMax, sum => brzSum}
import breeze.numerics.{log => brzLog}
import com.github.cloudml.zen.ml.util.Utils
import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.linalg.{DenseMatrix => SDM, DenseVector => SDV, Matrix => SM, SparseMatrix => SSM, SparseVector => SSV, Vector => SV}

@Experimental
object NNUtil {
  def initializeBias(numOut: Int): BDV[Double] = {
    BDV.zeros[Double](numOut)
  }

  def initializeWeight(numIn: Int, numOut: Int): BDM[Double] = {
    BDM.zeros[Double](numOut, numIn)
  }

  def initializeWeight(numIn: Int, numOut: Int, rand: () => Double): BDM[Double] = {
    val weight = initializeWeight(numIn, numOut)
    initializeWeight(weight, rand)
  }

  def initializeWeight(w: BDM[Double], rand: () => Double): BDM[Double] = {
    for (i <- 0 until w.rows) {
      for (j <- 0 until w.cols) {
        w(i, j) = rand()
      }
    }
    w
  }

  def initUniformDistWeight(numIn: Int, numOut: Int): BDM[Double] = {
    initUniformDistWeight(initializeWeight(numIn, numOut), 0.0)
  }

  def initUniformDistWeight(numIn: Int, numOut: Int, scale: Double): BDM[Double] = {
    initUniformDistWeight(initializeWeight(numIn, numOut), scale)
  }

  def initUniformDistWeight(w: BDM[Double], scale: Double): BDM[Double] = {
    val numIn = w.cols
    val numOut = w.rows
    val s = if (scale <= 0) math.sqrt(6D / (numIn + numOut)) else scale
    initUniformDistWeight(w, -s, s)
  }

  def initUniformDistWeight(numIn: Int, numOut: Int, low: Double, high: Double): BDM[Double] = {
    initUniformDistWeight(initializeWeight(numIn, numOut), low, high)
  }

  def initUniformDistWeight(w: BDM[Double], low: Double, high: Double): BDM[Double] = {
    initializeWeight(w, () => Utils.random.nextDouble() * (high - low) + low)
  }

  def initGaussianDistWeight(numIn: Int, numOut: Int): BDM[Double] = {
    initGaussianDistWeight(initializeWeight(numIn, numOut), 0.0)
  }

  def initGaussianDistWeight(numIn: Int, numOut: Int, scale: Double): BDM[Double] = {
    initGaussianDistWeight(initializeWeight(numIn, numOut), scale)
  }

  def initGaussianDistWeight(weight: BDM[Double], scale: Double): BDM[Double] = {
    val sd = if (scale <= 0) 0.01 else scale
    initializeWeight(weight, () => Utils.random.nextGaussian() * sd)
  }

  @inline def softplus(x: Double, expThreshold: Double = 64): Double = {
    if (x > expThreshold) {
      x
    } else if (x < -expThreshold) {
      0
    } else {
      math.log1p(math.exp(x))
    }
  }

  @inline def softplusPrimitive(y: Double, expThreshold: Double = 64): Double = {
    if (y > expThreshold) {
      1
    } else {
      val z = math.exp(y)
      (z - 1) / z
    }
  }

  @inline def tanh(x: Double): Double = {
    val a = math.pow(math.exp(x), 2)
    (a - 1) / (a + 1)
  }

  @inline def tanhPrimitive(y: Double): Double = {
    1 - math.pow(y, 2)
  }

  @inline def sigmoid(x: Double): Double = {
    1d / (1d + math.exp(-x))
  }

  @inline def sigmoid(x: Double, expThreshold: Double): Double = {
    if (x > expThreshold) {
      1D
    } else if (x < -expThreshold) {
      0D
    } else {
      sigmoid(x)
    }
  }

  @inline def sigmoidPrimitive(y: Double): Double = {
    y * (1 - y)
  }

  @inline def softMaxPrimitive(y: Double): Double = {
    y * (1 - y)
  }

  def scalarExp(x: Double, expThreshold: Double = 64D): Double = {
    if (x < -expThreshold) {
      math.exp(-expThreshold)
    } else if (x > expThreshold) {
      math.exp(-expThreshold)
    }
    else {
      math.exp(x)
    }
  }

  def meanSquaredError(out: BDM[Double], label: BDM[Double]): Double = {
    require(label.rows == out.rows)
    require(label.cols == out.cols)
    var diff = 0D
    for (i <- 0 until out.rows) {
      for (j <- 0 until out.cols) {
        diff += math.pow(label(i, j) - out(i, j), 2)
      }
    }
    diff
  }

  def crossEntropy(output: BDM[Double], label: BDM[Double]): Double = {
    require(label.rows == output.rows)
    require(label.cols == output.cols)
    0 - brzSum(label :* brzLog(output))
  }
}
