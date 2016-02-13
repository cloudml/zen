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

package com.github.cloudml.zen.ml.optimization

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.{DenseVector, Vector, Vectors}

/**
 * :: DeveloperApi ::
 * Class used to compute the gradient for a loss function, given a single data point.
 */
@DeveloperApi
abstract class Gradient extends Serializable {
  /**
   * Compute the gradient and loss given the features of a single data point.
   *
   * @param data features for one data point
   * @param label label for this data point
   * @param weights weights/coefficients corresponding to features
   *
   * @return (gradient: Vector, loss: Double)
   */
  def compute(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
    val gradient = Vectors.zeros(weights.size)
    val loss = compute(data, label, weights, gradient)
    (gradient, loss)
  }

  /**
   * Compute the gradient and loss given the features of a single data point,
   * add the gradient to a provided vector to avoid creating new objects, and return loss.
   *
   * @param data features for one data point
   * @param label label for this data point
   * @param weights weights/coefficients corresponding to features
   * @param cumGradient the computed gradient will be added to this vector
   *
   * @return loss
   */
  def compute(data: Vector, label: Double, weights: Vector, cumGradient: Vector): Double

  /**
   * Compute the gradient and loss given the iterator.
   *
   * @param iter Iterator for (label, data) pair
   * @param weights weights/coefficients corresponding to features
   * @param cumGradient the computed gradient will be added to this vector
   *
   * @return (count: Long, loss: Double)
   */
  def compute(
    iter: Iterator[(Double, Vector)],
    weights: Vector,
    cumGradient: Vector): (Long, Double) = {
    var loss = 0D
    var count = 0L
    iter.foreach { t =>
      loss += compute(t._2, t._1, weights, cumGradient)
      count += 1
    }
    (count, loss)
  }
}
