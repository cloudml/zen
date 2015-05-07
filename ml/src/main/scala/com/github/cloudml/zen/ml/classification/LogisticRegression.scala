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

import breeze.linalg.max
import breeze.numerics.{abs, signum, sqrt, exp}
import com.github.cloudml.zen.ml.util.Utils
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.{Logging}
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils._
import org.apache.spark.rdd.RDD
import com.github.cloudml.zen.ml.linalg.BLAS.dot
import com.github.cloudml.zen.ml.linalg.BLAS.axpy
import com.github.cloudml.zen.ml.linalg.BLAS.scal
import org.apache.spark.storage.StorageLevel

class LogisticRegressionMIS(dataSet: RDD[LabeledPoint]) extends Logging with Serializable{
  private var epsilon: Double = 1e-4
  private var stepSize: Double = 1.0
  private var numIterations: Int = 100
  private var regParam: Double = 0.0
  private var miniBatchFraction: Double = 1.0
  /**
   * In `GeneralizedLinearModel`, only single linear predictor is allowed for both weights
   * and intercept. However, for multinomial logistic regression, with K possible outcomes,
   * we are training K-1 independent binary logistic regression models which requires K-1 sets
   * of linear predictor.
   *
   * As a result, the workaround here is if more than two sets of linear predictors are needed,
   * we construct bigger `weights` vector which can hold both weights and intercepts.
   * If the intercepts are added, the dimension of `weights` will be
   * (numOfLinearPredictor) * (numFeatures + 1) . If the intercepts are not added,
   * the dimension of `weights` will be (numOfLinearPredictor) * numFeatures.
   *
   * Thus, the intercepts will be encapsulated into weights, and we leave the value of intercept
   * in GeneralizedLinearModel as zero.
   */
  protected var numOfLinearPredictor: Int = 1
  /** Whether to add intercept (default: false). */
  protected var addIntercept: Boolean = false
  /**
   * The dimension of training features.
   */
  protected var numFeatures: Int = -1
  /**
   * Whether to perform feature scaling before model training to reduce the condition numbers
   * which can significantly help the optimizer converging faster. The scaling correction will be
   * translated back to resulting model weights, so it's transparent to users.
   * Note: This technique is used in both libsvm and glmnet packages. Default false.
   */
  private var useFeatureScaling = false
  /**
   * Set if the algorithm should use feature scaling to improve the convergence during optimization.
   */
  private def setFeatureScaling(useFeatureScaling: Boolean): this.type = {
    this.useFeatureScaling = useFeatureScaling
    this
  }
  private val numSamples = dataSet.count()

  /**
   * Set Number of features
   * @param numFeatures
   * @return
   */
  def setNumFeatures(numFeatures: Int): this.type = {
    this.numFeatures = numFeatures
    this
  }
  /**
   * Set if the algorithm should add an intercept. Default false.
   * We set the default to false because adding the intercept will cause memory allocation.
   */
  def setIntercept(addIntercept: Boolean): this.type = {
    this.addIntercept = addIntercept
    this
  }
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
   * Run the algorithm with the configured parameters on an input
   * RDD of LabeledPoint entries.
   */
  def run(iterations: Int): (LogisticRegressionModel, Array[Double]) = {
    if (numFeatures < 0) {
      numFeatures = dataSet.map(_.features.size).first()
    }
    /**
     * When `numOfLinearPredictor > 1`, the intercepts are encapsulated into weights,
     * so the `weights` will include the intercepts. When `numOfLinearPredictor == 1`,
     * the intercept will be stored as separated value in `GeneralizedLinearModel`.
     * This will result in different behaviors since when `numOfLinearPredictor == 1`,
     * users have no way to set the initial intercept, while in the other case, users
     * can set the intercepts as part of weights.
     *
     * TODO: See if we can deprecate `intercept` in `GeneralizedLinearModel`, and always
     * have the intercept as part of weights to have consistent design.
     */
    val initialWeights = {
      if (numOfLinearPredictor == 1) {
        Vectors.dense(new Array[Double](numFeatures))
      } else if (addIntercept) {
        Vectors.dense(new Array[Double]((numFeatures + 1) * numOfLinearPredictor))
      } else {
        Vectors.dense(new Array[Double](numFeatures * numOfLinearPredictor))
      }
    }
    run(iterations, initialWeights)
  }
  def run(iterations: Int, initialWeights: Vector): (LogisticRegressionModel, Array[Double]) ={
    if (numFeatures < 0) {
      numFeatures = dataSet.map(_.features.size).first()
    }

    if (dataSet.getStorageLevel == StorageLevel.NONE) {
      logWarning("The input data is not directly cached, which may hurt performance if its"
        + " parent RDDs are also uncached.")
    }

    /*
     * Scaling columns to unit variance as a heuristic to reduce the condition number:
     *
     * During the optimization process, the convergence (rate) depends on the condition number of
     * the training dataset. Scaling the variables often reduces this condition number
     * heuristically, thus improving the convergence rate. Without reducing the condition number,
     * some training datasets mixing the columns with different scales may not be able to converge.
     *
     * GLMNET and LIBSVM packages perform the scaling to reduce the condition number, and return
     * the weights in the original scale.
     * See page 9 in http://cran.r-project.org/web/packages/glmnet/glmnet.pdf
     *
     * Here, if useFeatureScaling is enabled, we will standardize the training features by dividing
     * the variance of each column (without subtracting the mean), and train the model in the
     * scaled space. Then we transform the coefficients from the scaled space to the original scale
     * as GLMNET and LIBSVM do.
     *
     * Currently, it's only enabled in LogisticRegressionWithLBFGS
     */
//    val scaler = if (useFeatureScaling) {
//      new StandardScaler(withStd = true, withMean = false).fit(dataSet.map(_.features))
//    } else {
//      null
//    }
//    // Prepend an extra variable consisting of all 1.0's for the intercept.
//    // TODO: Apply feature scaling to the weight vector instead of input data.
//    val data =
//      if (addIntercept) {
//        if (useFeatureScaling) {
//          dataSet.map(lp => (lp.label, appendBias(scaler.transform(lp.features)))).cache()
//        } else {
//          dataSet.map(lp => (lp.label, appendBias(lp.features))).cache()
//        }
//      } else {
//        if (useFeatureScaling) {
//          dataSet.map(lp => (lp.label, scaler.transform(lp.features))).cache()
//        } else {
//          dataSet.map(lp => (lp.label, lp.features))
//        }
//      }

    /**
     * TODO: For better convergence, in logistic regression, the intercepts should be computed
     * from the prior probability distribution of the outcomes; for linear regression,
     * the intercept should be set as the average of response.
     */
    var initialWeightsWithIntercept = if (addIntercept && numOfLinearPredictor == 1) {
      appendBias(initialWeights)
    } else {
      /** If `numOfLinearPredictor > 1`, initialWeights already contains intercepts. */
      initialWeights
    }
    val lossArr = new Array[Double](iterations)
    for (iter <- 1 to iterations) {
      logInfo(s"Start train (Iteration $iter/$iterations)")
      val startedAt = System.nanoTime()
      val delta = backward(iter, forward(initialWeightsWithIntercept), numFeatures)
      initialWeightsWithIntercept = updateWeights(initialWeightsWithIntercept, delta, iter)
      val lossSum = loss(initialWeightsWithIntercept)
      lossArr(iter-1) = lossSum
      val elapsedSeconds = (System.nanoTime() - startedAt) / 1e9
      logInfo(s"train (Iteration $iter/$iterations) loss:              $lossSum")
      logInfo(s"End  train (Iteration $iter/$iterations) takes:         $elapsedSeconds")
    }
    val intercept = if (addIntercept && numOfLinearPredictor == 1) {
      initialWeightsWithIntercept(initialWeightsWithIntercept.size - 1)
    } else {
      0.0
    }
    val weights = if (addIntercept && numOfLinearPredictor == 1) {
      Vectors.dense(initialWeightsWithIntercept.toArray.slice(0, initialWeightsWithIntercept.size - 1))
    } else {
      initialWeightsWithIntercept
    }
    (new LogisticRegressionModel(weights, intercept), lossArr)
  }
  /**
   * Calculate the mistake probability: q(i) = 1/(1+exp(yi*(w*xi))).
   * @param initialWeights weights of last iteration.
   */
  protected[ml] def forward(initialWeights: Vector): RDD[Double] = {
    dataSet.map{point =>
      val z = point.label * dot(initialWeights, point.features)
      1.0 / (1.0 + exp(z))
    }
  }

  /**
   * Calculate the change in weights. delta_W_j = stepSize * log(mu_j_+/mu_j_-)
   * @param misProb q(i) = 1/(1+exp(yi*(w*xi))).
   */
  protected[ml] def backward(iter: Int, misProb: RDD[Double], numFeatures: Int): Vector = {
    def func(v1: Vector, v2: Vector) = {
      axpy(1.0, v1, v2)
      v2
    }
    dataSet.zip(misProb).map {
      case (point, prob) =>
        val scaledFeatures = point.features
        scal(prob, scaledFeatures)
        (point.label, scaledFeatures)
    }.aggregateByKey(Vectors.zeros(numFeatures))(func, func).reduce{ (x1, x2) =>
      val muPlus: Array[Double] = {if (x1._1 > 0) x1._2 else x2._2}.toArray
      val muMinus: Array[Double] = {if (x1._1 < 0) x1._2 else x2._2}.toArray
      assert(muPlus.length == muMinus.length)
      val grads: Array[Double] = new Array[Double](muPlus.length)
      var i = 0
      while (i < muPlus.length) {
        grads(i) = if (epsilon == 0.0) {
          math.log(muPlus(i) / muMinus(i))
        } else {
          math.log(epsilon + muPlus(i) / (epsilon + muMinus(i)))
        }
        i += 1
      }
      val thisIterStepSize = stepSize / math.sqrt(iter)
      val gradVec = Vectors.dense(grads)
      scal(thisIterStepSize, gradVec)
      (0.0, gradVec)
    }._2
  }

  /**
   * Update weights
   * @param weights
   * @param delta
   */
  protected[ml] def updateWeights(weights: Vector, delta: Vector, iter: Int): Vector = {
    axpy(1.0, delta, weights)
    val thisIterL1StepSize = stepSize / sqrt(iter)
    val newWeights = weights.toArray.map{ weight =>
        var newWeight = weight
        if (regParam > 0.0 && weight != 0.0) {
          val shrinkageVal = regParam * thisIterL1StepSize
          newWeight = signum(weight) * max(0.0, abs(weight) - shrinkageVal)
        }
        assert(!newWeight.isNaN)
        newWeight
    }
    Vectors.dense(newWeights)
  }
  /**
   * @param weights
   * @return Loss of given weights and dataSet in one iteration.
   */
  protected[ml] def loss(weights: Vector) : Double = {
    // For Binary Logistic Regression
    dataSet.map {point =>
      val margin = -1.0 * dot(point.features, weights)
      if (point.label > 0) {
        Utils.log1pExp(margin)
      } else {
        Utils.log1pExp(margin) - margin
      }
    }.reduce(_ + _) / numSamples
  }
}

object LogisticRegression {
  def trainMIS(
    input: RDD[LabeledPoint],
    numIterations: Int,
    stepSize: Double,
    regParam: Double,
    epsilon: Double = 1e-3,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK): (LogisticRegressionModel, Array[Double]) = {
    val lr = new LogisticRegressionMIS(input)
    lr.setEpsilon(epsilon)
      .setIntercept(false)
      .setStepSize(stepSize)
      .setNumIterations(numIterations)
      .setRegParam(regParam)
      .run(numIterations)
  }
}
