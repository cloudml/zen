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
import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import com.github.cloudml.zen.ml.linalg.BLAS.dot
import com.github.cloudml.zen.ml.linalg.BLAS.scal
import com.github.cloudml.zen.ml.linalg.BLAS.axpy

class LogisticRegressionMIS (
  private var stepSize: Double,
  private var numIterations: Int,
  private var regParm: Double,
  private var miniBatchFraction: Double)
extends Logging with Serializable{
  /**
   * Construct a LogisticRegression object with default parameters: {stepSize: 1.0,
   * numIterations: 100, regParm: 0.01, miniBatchFraction: 1.0}.
   */
  def this() = this(1.0, 100, 0.01, 1.0)

  private var epsilon = 1e-4

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
      val ywx = point.label * dot(initialWeights, point.features)
      1.0 / (1.0 + exp(ywx))
    }
  }

  /**
   * Calculate the change in weights. wj = log(mu_j_+/mu_j_-)
   * @param misProb q(i) = 1/(1+exp(yi*(w*xi))).
   * @param dataSet
   */
  protected[ml] def backward(misProb: RDD[Double], dataSet: RDD[LabeledPoint], numFeatures: Int):
  Array[Double] = {
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
    grads
  }
}
