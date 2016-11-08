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

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, SparseVector => BSV, argmax => brzArgMax, axpy => brzAxpy}
import com.github.cloudml.zen.ml.linalg.BLAS
import com.github.cloudml.zen.ml.util.{LoaderUtils, Logging, SparkUtils}
import com.github.cloudml.zen.ml.optimization._
import org.apache.spark.SparkContext
import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.linalg.{DenseVector => SDV, SparseVector => SSV, Vector => SV}
import org.apache.spark.mllib.util.Loader
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.json4s.DefaultFormats
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

@Experimental
object MLP extends Logging with Loader[MLPModel] {

  override def load(sc: SparkContext, path: String): MLPModel = {
    SaveLoadV1_0.load(sc, path)
  }

  def train(
    data: RDD[(SV, SV)],
    batchSize: Int,
    numIteration: Int,
    topology: Array[Int],
    fraction: Double,
    learningRate: Double,
    weightCost: Double): MLPModel = {
    train(data, batchSize, numIteration, new MLPModel(topology), fraction, learningRate, weightCost)
  }

  def train(
    data: RDD[(SV, SV)],
    batchSize: Int,
    numIteration: Int,
    nn: MLPModel,
    fraction: Double,
    learningRate: Double,
    weightCost: Double): MLPModel = {
    runSGD(data, nn, batchSize, numIteration, fraction, learningRate, weightCost)
  }

  def runSGD(
    trainingRDD: RDD[(SV, SV)],
    batchSize: Int,
    topology: Array[Int],
    maxNumIterations: Int,
    fraction: Double,
    learningRate: Double,
    weightCost: Double): MLPModel = {
    val nn = new MLPModel(topology)
    runSGD(trainingRDD, nn, batchSize, maxNumIterations, fraction, learningRate, weightCost)
  }

  def runSGD(
    data: RDD[(SV, SV)],
    nn: MLPModel,
    batchSize: Int,
    maxNumIterations: Int,
    fraction: Double,
    learningRate: Double,
    weightCost: Double): MLPModel = {
    val updater = new MLPEquilibratedUpdater(nn.topology, 1e-6, 1e-2, 0)
    runSGD(data, nn, updater, batchSize, maxNumIterations, fraction, learningRate, weightCost)
  }

  @Experimental
  def runSGD(
    data: RDD[(SV, SV)],
    mlp: MLPModel,
    updater: Updater,
    batchSize: Int,
    maxNumIterations: Int,
    fraction: Double,
    learningRate: Double,
    weightCost: Double): MLPModel = {
    val gradient = new MLPGradient(mlp.topology, mlp.innerLayers.map(_.layerType),
      mlp.dropout, batchSize)

    val optimizer = new GradientDescent(gradient, updater).
      setMiniBatchFraction(fraction).
      setNumIterations(maxNumIterations).
      setRegParam(weightCost).
      setStepSize(learningRate)

    val trainingRDD = toTrainingRDD(data)
    trainingRDD.persist(StorageLevel.MEMORY_AND_DISK).setName("MLP-dataBatch")
    val weights = optimizer.optimize(trainingRDD, toVector(mlp))
    trainingRDD.unpersist()
    fromVector(mlp, weights)
    mlp
  }


  def runLBFGS(
    trainingRDD: RDD[(SV, SV)],
    topology: Array[Int],
    batchSize: Int,
    maxNumIterations: Int,
    convergenceTol: Double,
    weightCost: Double): MLPModel = {
    val nn = new MLPModel(topology)
    runLBFGS(trainingRDD, nn, batchSize, maxNumIterations,
      convergenceTol, weightCost)
  }

  def runLBFGS(
    data: RDD[(SV, SV)],
    mlp: MLPModel,
    batchSize: Int,
    maxNumIterations: Int,
    convergenceTol: Double,
    weightCost: Double): MLPModel = {
    val gradient = new MLPGradient(mlp.topology, mlp.innerLayers.map(_.layerType), mlp.dropout, batchSize)
    val updater = new MLPUpdater(mlp.topology)
    val optimizer = new LBFGS(gradient, updater).
      setConvergenceTol(convergenceTol).
      setNumIterations(maxNumIterations).
      setRegParam(weightCost)

    val trainingRDD = toTrainingRDD(data)
    trainingRDD.persist(StorageLevel.MEMORY_AND_DISK).setName("MLP-dataBatch")
    val weights = optimizer.optimize(trainingRDD, toVector(mlp))
    trainingRDD.unpersist()
    fromVector(mlp, weights)
    mlp
  }

  private[ml] def fromVector(mlp: MLPModel, weights: SV): Unit = {
    val structure = vectorToStructure(mlp.topology, weights)
    val layers: Array[Layer] = mlp.innerLayers
    for (i <- structure.indices) {
      val (weight, bias) = structure(i)
      val layer = layers(i)
      layer.weight := weight
      layer.bias := bias
    }
  }

  private[ml] def toVector(nn: MLPModel): SV = {
    structureToVector(nn.innerLayers.map(l => (l.weight, l.bias)))
  }

  private[ml] def structureToVector(grads: Array[(BDM[Double], BDV[Double])]): SV = {
    val numLayer = grads.length
    val sumLen = grads.map(m => m._1.rows * m._1.cols + m._2.size).sum
    val data = new Array[Double](sumLen)
    var offset = 0
    for (l <- 0 until numLayer) {
      val (gradWeight, gradBias) = grads(l)
      val numIn = gradWeight.cols
      val numOut = gradWeight.rows
      Array.copy(gradWeight.toArray, 0, data, offset, numOut * numIn)
      offset += numIn * numOut
      Array.copy(gradBias.toArray, 0, data, offset, numOut)
      offset += numOut
    }
    new SDV(data)
  }

  private[ml] def vectorToStructure(
    topology: Array[Int],
    weights: SV): Array[(BDM[Double], BDV[Double])] = {
    val data = weights.toArray
    val numLayer = topology.length - 1
    val grads = new Array[(BDM[Double], BDV[Double])](numLayer)
    var offset = 0
    for (layer <- 0 until numLayer) {
      val numIn = topology(layer)
      val numOut = topology(layer + 1)
      val weight = new BDM[Double](numOut, numIn, data, offset)
      offset += numIn * numOut
      val bias = new BDV[Double](data, offset, 1, numOut)
      offset += numOut
      grads(layer) = (weight, bias)
    }
    grads
  }

  def error(data: RDD[(SV, SV)], nn: MLPModel, batchSize: Int): Double = {
    val count = data.count()
    val dataBatches = batchMatrix(data, batchSize, nn.numInput, nn.numOut)
    val sumError = dataBatches.map { case (x, y) =>
      val h = nn.predict(x).toDenseMatrix
      (0 until h.cols).map(i => {
        if (brzArgMax(y(::, i)) == brzArgMax(h(::, i))) 0D else 1D
      }).sum
    }.sum
    sumError / count
  }

  private def batchMatrix(
    data: RDD[(SV, SV)],
    batchSize: Int,
    numInput: Int,
    numOut: Int): RDD[(BDM[Double], BDM[Double])] = {
    val dataBatch = data.mapPartitions { itr =>
      itr.grouped(batchSize).map { seq =>
        val x = BDM.zeros[Double](numInput, seq.size)
        val y = BDM.zeros[Double](numOut, seq.size)
        seq.zipWithIndex.foreach { case (v, i) =>
          x(::, i) := SparkUtils.toBreeze(v._1)
          y(::, i) := SparkUtils.toBreeze(v._2)
        }
        (x, y)
      }
    }
    dataBatch
  }

  private def toTrainingRDD(data: RDD[(SV, SV)]): RDD[(Double, SV)] = {
    data.map { case (input, label) =>
      val len = input.size + label.size
      val data = if (input.isInstanceOf[SSV]) BSV.zeros[Double](len) else BDV.zeros[Double](len)
      data(0 until input.size) := SparkUtils.toBreeze(input)
      data(input.size until len) := SparkUtils.toBreeze(label)
      (0D, SparkUtils.fromBreeze(data))
    }
  }

  private[ml] def initLayers(topology: Array[Int]): Array[Layer] = {
    val numLayer = topology.length - 1
    val layers = new Array[Layer](numLayer)
    for (layer <- (0 until numLayer).reverse) {
      val numIn = topology(layer)
      val numOut = topology(layer + 1)
      layers(layer) = if (layer == numLayer - 1) {
        new SoftMaxLayer(numIn, numOut)
      }
      else {
        new ReLuLayer(numIn, numOut)
      }
      println(s"layers($layer) = $numIn * $numOut")
    }
    layers
  }

  private[ml] def initLayers(
    params: Array[(BDM[Double], BDV[Double])],
    layerTypes: Array[String]): Array[Layer] = {
    val numLayer = params.length
    val layers = new Array[Layer](numLayer)
    for (layer <- 0 until numLayer) {
      val (weight, bias) = params(layer)
      layers(layer) = Layer.initializeLayer(weight, bias, layerTypes(layer))
    }
    layers
  }

  private[ml] def initDropout(numLayer: Int, d: Array[Double]): Array[Double] = {
    require(d.length > 0)
    val dropout = new Array[Double](numLayer)
    for (layer <- 0 until numLayer) {
      dropout(layer) = if (layer == numLayer - 1) {
        0D
      } else if (layer < d.length) {
        d(layer)
      } else {
        d.last
      }
    }
    dropout
  }

  private[ml] def l2(
    topology: Array[Int],
    weightsOld: SV,
    gradient: SV,
    stepSize: Double,
    iter: Int,
    regParam: Double): Double = {
    if (regParam > 0D) {
      var norm = 0D
      val nn = MLP.vectorToStructure(topology, weightsOld)
      val grads = MLP.vectorToStructure(topology, gradient)
      for (layer <- nn.indices) {
        brzAxpy(regParam, nn(layer)._1, grads(layer)._1)
        for (i <- 0 until nn(layer)._1.rows) {
          for (j <- 0 until nn(layer)._1.cols) {
            norm += math.pow(nn(layer)._1(i, j), 2)
          }
        }
      }
      // TODO: why?
      0.5 * regParam * norm * norm
    } else {
      regParam
    }
  }

  private[ml] object SaveLoadV1_0 {
    val formatVersionV1_0 = "1.0"
    val classNameV1_0 = "com.github.cloudml.zen.ml.neuralNetwork.MLPModel"

    def load(sc: SparkContext, path: String): MLPModel = {
      val (loadedClassName, version, metadata) = LoaderUtils.loadMetadata(sc, path)
      val versionV1_0 = SaveLoadV1_0.formatVersionV1_0
      val classNameV1_0 = SaveLoadV1_0.classNameV1_0
      if (loadedClassName == classNameV1_0 && version == versionV1_0) {
        implicit val formats = DefaultFormats
        val topology = (metadata \ "topology").extract[String].split(",").map(_.toInt)
        val dropout = (metadata \ "dropout").extract[String].split(",").map(_.toDouble)
        val layerType = (metadata \ "layerType").extract[String].split(",")
        val dataPath = LoaderUtils.dataPath(path)
        val data = sc.objectFile[SV](dataPath).first()
        val structures = MLP.vectorToStructure(topology, data)
        val layers = layerType.indices.map { index =>
          val (weight, bias) = structures(index)
          Layer.initializeLayer(weight, bias, layerType(index))
        }
        new MLPModel(layers.toArray, dropout)
      } else {
        throw new Exception(
          s"MLP.load did not recognize model with (className, format version):" +
            s"($loadedClassName, $version).  Supported:\n" +
            s"  ($classNameV1_0, 1.0)")
      }

    }

    def save(
      sc: SparkContext,
      path: String,
      mlp: MLPModel): Unit = {
      val data = MLP.toVector(mlp)
      val topology = mlp.topology
      val dropout = mlp.dropout
      val layerType = mlp.innerLayers.map(_.layerType)
      val metadata = compact(render
      (("class" -> classNameV1_0) ~ ("version" -> formatVersionV1_0) ~
        ("topology" -> topology.mkString(",")) ~ ("dropout" -> dropout.mkString(",")) ~
        ("layerType" -> layerType.mkString(","))))
      sc.parallelize(Seq(metadata), 1).saveAsTextFile(LoaderUtils.metadataPath(path))
      sc.parallelize(Seq(data), 1).saveAsObjectFile(LoaderUtils.dataPath(path))
    }
  }

}

private[ml] class MLPGradient(
  val topology: Array[Int],
  val layerTypes: Array[String],
  val dropoutRate: Array[Double],
  val batchSize: Int) extends Gradient {

  override def compute(data: SV, label: Double, weights: SV): (SV, Double) = {
    val layers = MLP.initLayers(MLP.vectorToStructure(topology, weights), layerTypes)
    val mlp = new MLPModel(layers, dropoutRate)
    val numIn = mlp.numInput
    val numLabel = mlp.numOut
    assert(data.size == numIn + numLabel)
    val batchedData = data.toArray
    val input = new BDM(numIn, 1, batchedData.slice(0, numIn))
    val label = new BDM(numLabel, 1, batchedData.slice(numIn, numIn + numLabel))
    val (grads, error, _) = mlp.computeGradient(input, label)
    (MLP.structureToVector(grads), error)
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
    val layers = MLP.initLayers(MLP.vectorToStructure(topology, weights), layerTypes)
    val mlp = new MLPModel(layers, dropoutRate)
    val numIn = mlp.numInput
    val numLabel = mlp.numOut
    var loss = 0D
    var count = 0L
    iter.map(_._2).grouped(batchSize).foreach { seq =>
      val numCol = seq.size
      val input = BDM.zeros[Double](numIn, numCol)
      val label = BDM.zeros[Double](numLabel, numCol)

      seq.zipWithIndex.foreach { case (data, index) =>
        assert(data.size == numIn + numLabel)
        val brzVector = SparkUtils.toBreeze(data)
        input(::, index) := brzVector(0 until numIn)
        label(::, index) := brzVector(numIn until numIn + numLabel)
      }
      val (grads, error, _) = mlp.computeGradient(input, label)
      BLAS.axpy(1, MLP.structureToVector(grads), cumGradient)
      loss += error
      count += numCol
    }
    (count, loss)
  }
}

private[ml] class MLPUpdater(val topology: Array[Int]) extends Updater {
  override def compute(
    weightsOld: SV,
    gradient: SV,
    stepSize: Double,
    iter: Int,
    regParam: Double): (SV, Double) = {
    val newReg = MLP.l2(topology, weightsOld, gradient, stepSize, iter, regParam)
    BLAS.axpy(-stepSize, gradient, weightsOld)
    (weightsOld, newReg)
  }
}

@Experimental
class MLPAdaGradUpdater(
  val topology: Array[Int],
  rho: Double = 1 - 1e-2,
  epsilon: Double = 1e-2,
  gamma: Double = 1e-1,
  momentum: Double = 0.0) extends AdaGradUpdater(rho, epsilon, gamma, momentum) {
  override protected def l2(
    weightsOld: SV,
    gradient: SV,
    stepSize: Double,
    iter: Int,
    regParam: Double): Double = {
    MLP.l2(topology, weightsOld, gradient, stepSize, iter, regParam)
  }
}

@Experimental
class MLPEquilibratedUpdater(
  val topology: Array[Int],
  _epsilon: Double = 1e-6,
  _gamma: Double = 1e-2,
  _momentum: Double = 0.0) extends EquilibratedUpdater(_epsilon, _gamma, _momentum) {
  override protected def l2(
    weightsOld: SV,
    gradient: SV,
    stepSize: Double,
    iter: Int,
    regParam: Double): Double = {
    MLP.l2(topology, weightsOld, gradient, stepSize, iter, regParam)
  }
}

@Experimental
class MLPAdaDeltaUpdater(
  val topology: Array[Int],
  rho: Double = 0.99,
  epsilon: Double = 1e-8,
  momentum: Double = 0.0) extends AdaDeltaUpdater(rho, epsilon, momentum) {
  override protected def l2(
    weightsOld: SV,
    gradient: SV,
    stepSize: Double,
    iter: Int,
    regParam: Double): Double = {
    MLP.l2(topology, weightsOld, gradient, stepSize, iter, regParam)
  }
}

@Experimental
class MLPMomentumUpdater(
  val topology: Array[Int],
  momentum: Double = 0.9) extends MomentumUpdater(momentum) {
  override protected def l2(
    weightsOld: SV,
    gradient: SV,
    stepSize: Double,
    iter: Int,
    regParam: Double): Double = {
    MLP.l2(topology, weightsOld, gradient, stepSize, iter, regParam)
  }
}
