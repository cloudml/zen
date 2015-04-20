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
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import com.github.cloudml.zen.ml.linalg.BLAS.dot

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

  /**
   * Calculate the mistake probability: q(i) = 1/(1+exp(yi*(w*xi))).
   * @param initialWeights weights of last iteration.
   * @param dataSet
   */
  protected[ml] def forward(initialWeights: Vector, dataSet: RDD[LabeledPoint]): RDD[(Double,
    Double)] = {
    dataSet.map{point =>
      val ywx = point.label * dot(initialWeights, point.features)
      (point.label, 1.0 / (1.0 + exp(ywx)))
    }
  }

  /**
   * Calculate the change in weights. wj = log(mu_j_+/mu_j_-)
   * @param misProb q(i) = 1/(1+exp(yi*(w*xi))).
   * @param dataSet
   */
  protected[ml] def backward(misProb: RDD[(Double, Double)], dataSet: RDD[LabeledPoint]) = {

  }
}
