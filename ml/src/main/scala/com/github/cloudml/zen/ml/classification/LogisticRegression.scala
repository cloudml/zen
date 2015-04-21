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
package com.github.cloudml.zen.ml.classification

import breeze.numerics.exp
import com.github.cloudml.zen.ml.util.Utils
import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import com.github.cloudml.zen.ml.linalg.BLAS.dot
import com.github.cloudml.zen.ml.linalg.BLAS.axpy
import com.github.cloudml.zen.ml.linalg.BLAS.scal

class LogisticRegressionMIS extends Logging with Serializable{
  /**
   * Construct a LogisticRegression object with default parameters: {stepSize: 1.0,
   * numIterations: 100, regParm: 0.01, miniBatchFraction: 1.0}.
   */
  def this() = this(1.0, 100, 0.01, 1.0)

  private var epsilon: Double = 1e-4
  private var stepSize: Double = 1.0
  private var numIterations: Int = 100
  private var regParam: Double = 0.0
  private var miniBatchFraction: Double = 1.0
  /**
   * Set the initial step size of SGD for the first step. Default 1.0.
   * In subsequent steps, the step size will decrease with stepSize/sqrt(t)
   */
  def setStepSize(stepSize: Double): this.type = {
    this.stepSize = stepSize
    this
  }
  /**
   * Set fraction of data to be used for each SGD iteration.
   * Default 1.0 (corresponding to deterministic/classical gradient descent)
   */
  def setMiniBatchFraction(fraction: Double): this.type = {
    this.miniBatchFraction = fraction
    this
  }

  /**
   * Set the number of iterations for SGD. Default 100.
   */
  def setNumIterations(iters: Int): this.type = {
    this.numIterations = iters
    this
  }

  /**
   * Set the regularization parameter. Default 0.0.
   */
  def setRegParam(regParam: Double): this.type = {
    this.regParam = regParam
    this
  }
  /**
   * Set smooth parameter.
   * @param eps parameter for smooth, default 1e-4.
   * @return
   */
  def setEpsilon(eps: Double): this.type = {
    epsilon = eps
    this
  }
  /**
   * Calculate the mistake probability: q(i) = 1/(1+exp(yi*(w*xi))).
   * @param initialWeights weights of last iteration.
   * @param dataSet
   */
  protected[ml] def forward(initialWeights: Vector, dataSet: RDD[LabeledPoint]): RDD[Double] = {
    dataSet.map{point =>
      val z = point.label * dot(initialWeights, point.features)
      1.0 / (1.0 + exp(z))
    }
  }

  /**
   * Calculate the change in weights. wj = log(mu_j_+/mu_j_-)
   * @param misProb q(i) = 1/(1+exp(yi*(w*xi))).
   * @param dataSet
   */
  protected[ml] def backward(misProb: RDD[Double], dataSet: RDD[LabeledPoint], numFeatures: Int):
  Vector = {
    def func(v1: Vector, v2: Vector) = {
      axpy(1.0, v1, v2)
      v2
    }
    val muArr: Array[(Double, Vector)] = dataSet.zip(misProb).map {
      case (point, prob) =>
        val scaledFeatures = Vectors.zeros(numFeatures)
        axpy(prob, point.features, scaledFeatures)
        (point.label, scaledFeatures)
    }.aggregateByKey(Vectors.zeros(numFeatures))(func, func).collect()
    assert(muArr.length == 2)
    val grads: Array[Double] = new Array[Double](numFeatures)
    val muPlus: Array[Double] = {if (muArr(0)._1 > 0) muArr(0)._2 else muArr(1)._2}.toArray
    val muMinus: Array[Double] = {if (muArr(0)._1 < 0) muArr(0)._2 else muArr(1)._2}.toArray
    var i = 0
    while (i < numFeatures) {
      grads(i) = if (epsilon == 0.0) {
        math.log(muPlus(i) / muMinus(i))
      } else {
        math.log(muPlus(i) / (epsilon + muMinus(i)))
      }
      i += 1
    }
    Vectors.dense(grads)
  }

  /**
   * delta = stepSize * grad
   * @param iter
   * @param grads
   */
  protected[ml] def updateGradients(iter: Int, grads: Vector): Unit = {
    val thisIterStepSize = stepSize / math.sqrt(iter)
    scal(thisIterStepSize, grads)
  }

  /**
   * Update weights
   * @param initialWeights
   * @param delta
   */
  protected[ml] def updateWeights(initialWeights: Vector, delta: Vector): Unit = {
    axpy(1.0, delta, initialWeights)
  }
  /**
   * @param weights
   * @param dataSet
   * @return Loss of given weights and dataSet in one iteration.
   */
  protected[ml] def loss(weights: Vector, dataSet: RDD[LabeledPoint]) : Double = {
    // For Binary Logistic Regression
    var lossSum = 0
    dataSet.foreach {point =>
      val margin = -1.0 * dot(point.features, weights)
      if (point.label > 0) {
        lossSum += Utils.log1pExp(margin)
      } else {
        lossSum += Utils.log1pExp(margin) - margin
      }
    }
    lossSum
  }
}
