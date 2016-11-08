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

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, axpy => brzAxpy}
import com.github.cloudml.zen.ml.linalg.BLAS
import com.github.cloudml.zen.ml.util._
import com.github.cloudml.zen.ml.optimization._
import org.apache.spark.mllib.util.Loader
import org.apache.spark.SparkContext
import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.linalg.{DenseVector => SDV, Vector => SV}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.json4s.DefaultFormats
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

@Experimental
object RBM extends Logging with Loader[RBMModel] {

  override def load(sc: SparkContext, path: String): RBMModel = {
    SaveLoadV1_0.load(sc, path)
  }

  def train(
    data: RDD[SV],
    batchSize: Int,
    numIteration: Int,
    numVisible: Int,
    numHidden: Int,
    fraction: Double,
    learningRate: Double,
    weightCost: Double): RBMModel = {
    val rbm = new RBMModel(numVisible, numHidden)
    train(data, batchSize, numIteration, rbm, fraction, learningRate, weightCost)
  }

  def train(
    data: RDD[SV],
    batchSize: Int,
    numIteration: Int,
    rbm: RBMModel,
    fraction: Double,
    learningRate: Double,
    weightCost: Double): RBMModel = {
    runSGD(data, rbm, batchSize, numIteration, fraction, learningRate, weightCost)
  }

  def runSGD(
    trainingRDD: RDD[SV],
    batchSize: Int,
    numVisible: Int,
    numHidden: Int,
    maxNumIterations: Int,
    fraction: Double,
    learningRate: Double,
    weightCost: Double): RBMModel = {
    val rbm = new RBMModel(numVisible, numHidden)
    runSGD(trainingRDD, rbm, batchSize, maxNumIterations, fraction, learningRate, weightCost)
  }

  def runSGD(
    data: RDD[SV],
    rbm: RBMModel,
    batchSize: Int,
    maxNumIterations: Int,
    fraction: Double,
    learningRate: Double,
    weightCost: Double): RBMModel = {
    val numVisible = rbm.numIn
    val numHidden = rbm.numOut
    val updater = new RBMAdaGradUpdater(numVisible, numHidden, 0D, 1E-6)
    runSGD(data, rbm, updater, batchSize, maxNumIterations, fraction, learningRate, weightCost)
  }

  @Experimental
  def runSGD(
    data: RDD[SV],
    rbm: RBMModel,
    updater: Updater,
    batchSize: Int,
    maxNumIterations: Int,
    fraction: Double,
    learningRate: Double,
    weightCost: Double): RBMModel = {
    val gradient = new RBMGradient(rbm.numIn, rbm.numOut, rbm.dropoutRate, batchSize,
      rbm.visibleLayerType, rbm.hiddenLayerType)
    val optimizer = new GradientDescent(gradient, updater).
      setMiniBatchFraction(fraction).
      setNumIterations(maxNumIterations).
      setRegParam(weightCost).
      setStepSize(learningRate)
    val trainingRDD = data.map(t => (0D, t))
    // TODO: the related jira SPARK-4526
    trainingRDD.persist(StorageLevel.MEMORY_AND_DISK).setName("RBM-dataBatch")
    val weights = optimizer.optimize(trainingRDD, toVector(rbm))
    trainingRDD.unpersist()
    fromVector(rbm, weights)
    rbm
  }

  private[ml] def fromVector(rbm: RBMModel, weights: SV): Unit = {
    val (weight, visibleBias, hiddenBias) = vectorToStructure(rbm.numIn, rbm.numOut, weights)
    rbm.weight := weight
    rbm.visibleBias := visibleBias
    rbm.hiddenBias := hiddenBias
  }

  private[ml] def toVector(rbm: RBMModel): SV = {
    structureToVector(rbm.weight, rbm.visibleBias, rbm.hiddenBias)
  }

  private[ml] def structureToVector(
    weight: BDM[Double],
    visibleBias: BDV[Double],
    hiddenBias: BDV[Double]): SV = {
    val numVisible = visibleBias.size
    val numHidden = hiddenBias.size
    val sumLen = numHidden * numVisible + numVisible + numHidden
    val data = new Array[Double](sumLen)
    var offset = 0

    Array.copy(weight.toArray, 0, data, offset, numHidden * numVisible)
    offset += numHidden * numVisible

    Array.copy(visibleBias.toArray, 0, data, offset, numVisible)
    offset += numVisible

    Array.copy(hiddenBias.toArray, 0, data, offset, numHidden)
    offset += numHidden

    new SDV(data)
  }

  private[ml] def vectorToStructure(
    numVisible: Int,
    numHidden: Int,
    weights: SV): (BDM[Double], BDV[Double], BDV[Double]) = {
    val data = weights.toArray
    var offset = 0

    val weight = new BDM[Double](numHidden, numVisible, data, offset)
    offset += numHidden * numVisible

    val visibleBias = new BDV[Double](data, offset, 1, numVisible)
    offset += numVisible

    val hiddenBias = new BDV[Double](data, offset, 1, numHidden)
    offset += numHidden

    (weight, visibleBias, hiddenBias)
  }

  private[ml] def l2(
    numVisible: Int,
    numHidden: Int,
    weightsOld: SV,
    gradient: SV,
    stepSize: Double,
    iter: Int,
    regParam: Double): Double = {
    if (regParam > 0D) {
      val (weight, _, _) = RBM.vectorToStructure(numVisible, numHidden, weightsOld)
      val (gradWeight, _, _) = RBM.vectorToStructure(numVisible, numHidden, gradient)
      brzAxpy(regParam, weight, gradWeight)
      var norm = 0D
      for (i <- 0 until weight.rows) {
        for (j <- 0 until weight.cols) {
          norm += math.pow(weight(i, j), 2)
        }
      }
      0.5 * regParam * norm * norm
    } else {
      regParam
    }
  }

  private[ml] object SaveLoadV1_0 {
    val formatVersionV1_0 = "1.0"
    val classNameV1_0 = "com.github.cloudml.zen.ml.neuralNetwork.RBMModel"

    def load(sc: SparkContext, path: String): RBMModel = {
      val (loadedClassName, version, metadata) = LoaderUtils.loadMetadata(sc, path)
      val versionV1_0 = SaveLoadV1_0.formatVersionV1_0
      val classNameV1_0 = SaveLoadV1_0.classNameV1_0
      if (loadedClassName == classNameV1_0 && version == versionV1_0) {
        implicit val formats = DefaultFormats
        val dropoutRate = (metadata \ "dropoutRate").extract[Double]
        val visibleLayerType = (metadata \ "visibleLayerType").extract[String]
        val hiddenLayerType = (metadata \ "hiddenLayerType").extract[String]
        val numVisible = (metadata \ "numVisible").extract[Int]
        val numHidden = (metadata \ "numHidden").extract[Int]
        val dataPath = LoaderUtils.dataPath(path)
        val data = sc.objectFile[SV](dataPath).first()
        val (weight, visibleBias, hiddenBias) = RBM.vectorToStructure(numVisible, numHidden, data)
        new RBMModel(weight, visibleBias, hiddenBias, dropoutRate, visibleLayerType, hiddenLayerType)
      } else {
        throw new Exception(
          s"RBM.load did not recognize model with (className, format version):" +
            s"($loadedClassName, $version).  Supported:\n" +
            s"  ($classNameV1_0, 1.0)")
      }

    }

    def save(
      sc: SparkContext,
      path: String,
      rbm: RBMModel): Unit = {
      val data = RBM.toVector(rbm)
      val numVisible: Int = rbm.numIn
      val numHidden: Int = rbm.numOut
      val dropoutRate = rbm.dropoutRate
      val visibleLayerType: String = rbm.visibleLayerType
      val hiddenLayerType: String = rbm.hiddenLayerType
      val metadata = compact(render
      (("class" -> classNameV1_0) ~ ("version" -> formatVersionV1_0) ~
        ("dropoutRate" -> dropoutRate) ~ ("visibleLayerType" -> visibleLayerType) ~
        ("hiddenLayerType" -> hiddenLayerType) ~ ("numVisible" -> numVisible) ~ ("numHidden" -> numHidden)))
      sc.parallelize(Seq(metadata), 1).saveAsTextFile(LoaderUtils.metadataPath(path))
      sc.parallelize(Seq(data), 1).saveAsObjectFile(LoaderUtils.dataPath(path))
    }
  }

}

private[ml] class RBMGradient(
  val numIn: Int,
  val numOut: Int,
  val dropoutRate: Double,
  val batchSize: Int,
  val visibleLayerType: String,
  val hiddenLayerType: String) extends Gradient {
  override def compute(data: SV, label: Double, weights: SV): (SV, Double) = {
    val (weight, visibleBias, hiddenBias) = RBM.vectorToStructure(numIn, numOut, weights)
    val rbm = new RBMModel(weight, visibleBias, hiddenBias, dropoutRate, visibleLayerType, hiddenLayerType)
    val input = new BDM(numIn, 1, data.toArray)
    val (gradWeight, gradVisibleBias, gradHiddenBias, error, _) = rbm.learn(input)
    (RBM.structureToVector(gradWeight, gradVisibleBias, gradHiddenBias), error)
  }

  override def compute(
    data: SV,
    label: Double,
    weights: SV,
    cumGradient: SV): Double = {
    val (grad, err) = compute(data, label, weights)
    BLAS.axpy(1, grad, cumGradient)
    err
  }

  override def compute(
    iter: Iterator[(Double, SV)],
    weights: SV,
    cumGradient: SV): (Long, Double) = {
    val (weight, visibleBias, hiddenBias) = RBM.vectorToStructure(numIn, numOut, weights)
    val rbm = new RBMModel(weight, visibleBias, hiddenBias, dropoutRate, visibleLayerType, hiddenLayerType)
    var loss = 0D
    var count = 0L
    iter.map(_._2).grouped(batchSize).foreach { seq =>
      val numCol = seq.size
      val input: BDM[Double] = BDM.zeros(numIn, numCol)
      seq.zipWithIndex.foreach { case (data, index) =>
        assert(data.size == numIn)
        input(::, index) := SparkUtils.toBreeze(data)
      }
      var (gradWeight, gradVisibleBias,
      gradHiddenBias, error, _) = rbm.learn(input)
      val w = RBM.structureToVector(gradWeight, gradVisibleBias, gradHiddenBias)
      BLAS.axpy(1, w, cumGradient)
      loss += error
      count += numCol
    }
    (count, loss)
  }
}

@Experimental
class RBMAdaGradUpdater(
  val numIn: Int,
  val numOut: Int,
  rho: Double = 0,
  epsilon: Double = 1e-6,
  gamma: Double = 1e-1,
  momentum: Double = 0.0) extends AdaGradUpdater(rho, epsilon, gamma, momentum) {

  override protected def l2(
    weightsOld: SV,
    gradient: SV,
    stepSize: Double,
    iter: Int,
    regParam: Double): Double = {
    RBM.l2(numIn, numOut, weightsOld, gradient, stepSize, iter, regParam)
  }
}

@Experimental
class RBMAdaDeltaUpdater(
  val numIn: Int,
  val numOut: Int,
  rho: Double = 0.95,
  epsilon: Double = 1e-6,
  momentum: Double = 0.0) extends AdaDeltaUpdater(rho, epsilon, momentum) {

  override protected def l2(
    weightsOld: SV,
    gradient: SV,
    stepSize: Double,
    iter: Int,
    regParam: Double): Double = {
    RBM.l2(numIn, numOut, weightsOld, gradient, stepSize, iter, regParam)
  }
}

@Experimental
class RBMMomentumUpdater(
  val numIn: Int,
  val numOut: Int,
  momentum: Double = 0.9) extends MomentumUpdater(momentum) {
  override protected def l2(
    weightsOld: SV,
    gradient: SV,
    stepSize: Double,
    iter: Int,
    regParam: Double): Double = {
    RBM.l2(numIn, numOut, weightsOld, gradient, stepSize, iter, regParam)
  }
}
