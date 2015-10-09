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

import com.github.cloudml.zen.ml.linalg.BLAS
import com.github.cloudml.zen.ml.util.SparkUtils._
import com.github.cloudml.zen.ml.util.Utils
import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.linalg.{Vector => SV, DenseVector => SDV, Vectors}
import com.github.cloudml.zen.ml.optimization._

/**
 * Equilibrated Gradient Descent the paper:
 * RMSProp and equilibrated adaptive learning rates for non-convex optimization
 * @param epsilon
 * @param momentum
 */
@Experimental
class EquilibratedUpdater(
  val epsilon: Double,
  val gamma: Double,
  val momentum: Double) extends Updater {
  require(momentum >= 0 && momentum < 1)
  @transient private var etaSum: SDV = null
  @transient private var momentumSum: SDV = null

  protected def l2(
    weightsOld: SV,
    gradient: SV,
    stepSize: Double,
    iter: Int,
    regParam: Double): Double = {
    0D
  }

  override def compute(
    weightsOld: SV,
    gradient: SV,
    stepSize: Double,
    iter: Int,
    regParam: Double): (SV, Double) = {
    if (etaSum == null) etaSum = new SDV(new Array[Double](weightsOld.size))
    val reg = l2(weightsOld, gradient, stepSize, iter, regParam)

    val grad = toBreeze(gradient)
    val e = toBreeze(etaSum)
    for (i <- 0 until grad.length) {
      e(i) += math.pow(grad(i) * Utils.random.nextGaussian(), 2)
    }

    etaSum.synchronized {
      for (i <- 0 until grad.length) {
        grad(i) = gamma * grad(i) / (epsilon + math.sqrt(etaSum(i) / iter))
      }
    }

    if (momentum > 0) {
      if (momentumSum == null) momentumSum = new SDV(new Array[Double](weightsOld.size))
      momentumSum.synchronized {
        BLAS.axpy(momentum, momentumSum, gradient)
        BLAS.copy(gradient, momentumSum)
      }
    }

    BLAS.axpy(-stepSize, gradient, weightsOld)
    (weightsOld, reg)
  }
}
