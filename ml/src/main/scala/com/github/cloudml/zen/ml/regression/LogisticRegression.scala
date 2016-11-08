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

package com.github.cloudml.zen.ml.regression

import scala.math._

import com.github.cloudml.zen.ml.partitioner.DBHPartitioner
import com.github.cloudml.zen.ml.util.SparkUtils._
import com.github.cloudml.zen.ml.util.{Logging, Utils}
import org.apache.spark.annotation.Experimental
import org.apache.spark.graphx2._
import org.apache.spark.graphx2.impl.{EdgeRDDImpl, GraphImpl}
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.linalg.{DenseVector => SDV}
import org.apache.spark.mllib.regression.{GeneralizedLinearModel, LabeledPoint}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import LogisticRegression._

abstract class LogisticRegression(
  @transient var dataSet: Graph[VD, ED],
  var stepSize: Double,
  var regParam: Double,
  var useAdaGrad: Boolean,
  @transient var storageLevel: StorageLevel) extends Serializable with Logging {

  def this(
    input: RDD[(VertexId, LabeledPoint)],
    stepSize: Double = 1e-4,
    regParam: Double = 0.0,
    useAdaGrad: Boolean = false,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK) {
    this(initializeDataSet(input, storageLevel), stepSize, regParam, useAdaGrad, storageLevel)
  }

  @transient protected var innerIter = 1
  @transient protected var checkpointInterval = 10

  def setStepSize(stepSize: Double): this.type = {
    this.stepSize = stepSize
    this
  }

  def setRegParam(regParam: Double): this.type = {
    this.regParam = regParam
    this
  }

  def setAdaGrad(useAdaGrad: Boolean): this.type = {
    this.useAdaGrad = useAdaGrad
    this
  }

  def setCheckpointInterval(interval: Int): this.type = {
    this.checkpointInterval = interval
    this
  }

  // ALL
  @transient protected var margin: VertexRDD[Double] = null
  @transient protected var gradientSum: VertexRDD[Double] = null
  @transient protected var deltaSum: VertexRDD[Array[Double]] = null
  @transient protected var gradient: VertexRDD[Double] = null
  @transient protected var vertices = dataSet.vertices
  @transient protected var previousVertices = vertices
  @transient protected var edges = dataSet.edges.asInstanceOf[EdgeRDDImpl[ED, _]].mapEdgePartitions {
    (pid, part) =>
      part.withoutVertexAttributes[VD]
  }.setName("edges").persist(storageLevel)
  if (edges.sparkContext.getCheckpointDir.isDefined) {
    edges.checkpoint()
    edges.count()
  }
  dataSet.edges.unpersist(blocking = false)
  if (vertices.getStorageLevel == StorageLevel.NONE) {
    vertices.persist(storageLevel)
    vertices.count()
  }
  dataSet = GraphImpl.fromExistingRDDs(vertices, edges)
  dataSet.persist(storageLevel)

  val numFeatures: Long = features.count()
  val numSamples: Long = samples.count()

  def samples: VertexRDD[VD] = {
    dataSet.vertices.filter(t => t._1 < 0)
  }

  def features: VertexRDD[VD] = {
    dataSet.vertices.filter(t => t._1 >= 0)
  }

  def run(iterations: Int): Unit = {
    for (iter <- 1 to iterations) {
      logInfo(s"Start train (Iteration $iter/$iterations)")
      val startedAt = System.nanoTime()
      previousVertices = dataSet.vertices
      margin = forward(iter)
      gradient = backward(margin, iter)
      gradient = updateGradientSum(gradient, iter)
      // gradient = updateDeltaSum(gradient, iter)

      val tis = thisIterStepSize(iter)
      vertices = updateWeight(gradient, iter, tis, stepSize / sqrt(iter))
      checkpointVertices()
      vertices.count()
      dataSet = GraphImpl.fromExistingRDDs(vertices, edges)
      val elapsedSeconds = (System.nanoTime() - startedAt) / 1e9
      // logInfo(s"train (Iteration $iter/$iterations) loss:              ${loss(margin)}")
      logInfo(s"End  train (Iteration $iter/$iterations) takes:         $elapsedSeconds")
      unpersistVertices()
      innerIter += 1
    }
  }

  protected def thisIterStepSize(iter: Int): Double = {
    if (useAdaGrad) {
      stepSize * min(iter / 11.0, 1.0)
    } else {
      stepSize / sqrt(iter)
    }
  }

  protected def forward(iter: Int): VertexRDD[VD]

  protected def backward(q: VertexRDD[VD], iter: Int): VertexRDD[Double]

  protected def loss(q: VertexRDD[VD]): Double

  def saveModel(numFeatures: Int = -1): GeneralizedLinearModel = {
    val len = if (numFeatures < 0) features.map(_._1).max().toInt + 1 else numFeatures
    val featureData = new Array[Double](len)
    features.toLocalIterator.foreach { case (index, value) =>
      featureData(index.toInt) = value
    }
    new LogisticRegressionModel(new SDV(featureData), 0.0, len, 2).clearThreshold()
  }

  protected def updateGradientSum(gradient: VertexRDD[Double], iter: Int): VertexRDD[Double] = {
    if (gradient.getStorageLevel == StorageLevel.NONE) {
      gradient.setName(s"gradient-$iter").persist(storageLevel)
    }
    if (useAdaGrad) {
      val delta = adaGrad(gradientSum, gradient, 1e-4, 1.0)
      checkpointGradientSum(delta)
      delta.setName(s"delta-$iter").persist(storageLevel).count()

      gradient.unpersist(blocking = false)
      val newGradient = delta.mapValues(_.head).setName(s"gradient-$iter").persist(storageLevel)
      newGradient.count()

      if (gradientSum != null) gradientSum.unpersist(blocking = false)
      gradientSum = delta.mapValues(_.last).setName(s"deltaSum-$iter").persist(storageLevel)
      gradientSum.count()
      delta.unpersist(blocking = false)
      newGradient
    } else {
      gradient
    }
  }

  protected def updateDeltaSum(gradient: VertexRDD[Double], iter: Int): VertexRDD[Double] = {
    if (gradient.getStorageLevel == StorageLevel.NONE) {
      gradient.setName(s"gradient-$iter").persist(storageLevel)
    }
    if (useAdaGrad) {
      val delta = adaDelta(deltaSum, gradient, 1e-8, 0.9)
      delta.setName(s"delta-$iter").persist(storageLevel).count()
      gradient.unpersist(blocking = false)
      val newGradient = delta.mapValues(_.head).setName(s"gradient-$iter").persist(storageLevel)
      newGradient.count()
      if (deltaSum != null) deltaSum.unpersist(blocking = false)
      deltaSum = delta.mapValues(_.tail).setName(s"deltaSum-$iter").persist(storageLevel)
      deltaSum.count()
      delta.unpersist(blocking = false)
      newGradient
    } else {
      gradient
    }
  }

  // Updater for L1 regularized problems
  protected def updateWeight(
    delta: VertexRDD[Double],
    iter: Int,
    thisIterStepSize: Double,
    thisIterL1StepSize: Double): VertexRDD[Double] = {
    dataSet.vertices.leftJoin(delta) { (_, attr, gradient) =>
      gradient match {
        case Some(gard) => {
          var weight = attr
          weight -= thisIterStepSize * gard
          if (regParam > 0.0 && weight != 0.0) {
            val shrinkageVal = regParam * thisIterL1StepSize
            weight = signum(weight) * max(0.0, abs(weight) - shrinkageVal)
          }
          assert(!weight.isNaN)
          weight
        }
        case None => attr
      }
    }.setName(s"vertices-$iter").persist(storageLevel)
  }

  protected def adaGrad(
    gradientSum: VertexRDD[Double],
    gradient: VertexRDD[Double],
    epsilon: Double,
    rho: Double): VertexRDD[Array[Double]] = {
    val delta = if (gradientSum == null) {
      gradient.mapValues(t => 0.0)
    }
    else {
      gradientSum
    }
    delta.innerJoin(gradient) { (_, gradSum, grad) =>
      val newGradSum = gradSum * rho + pow(grad, 2)
      val newGrad = grad / (epsilon + sqrt(newGradSum))
      Array(newGrad, newGradSum)
    }
  }

  protected def adaDelta(
    deltaSum: VertexRDD[Array[Double]],
    gradient: VertexRDD[Double],
    epsilon: Double,
    rho: Double): VertexRDD[Array[Double]] = {
    val delta = if (deltaSum == null) {
      gradient.mapValues(t => Array(0.0, 0.0))
    }
    else {
      deltaSum
    }
    delta.innerJoin(gradient) { case (_, Array(ds, gs), g) =>
      val ngs = rho * gs + (1 - rho) * pow(g, 2)
      val rms = sqrt(ds + epsilon) / sqrt(ngs + epsilon)
      val ng = rms * g
      val nds = rho * ds + (1 - rho) * pow(ng, 2)
      Array(ng, nds, ngs)
    }
  }

  protected def checkpointVertices(): Unit = {
    val sc = vertices.sparkContext
    if (innerIter % checkpointInterval == 0 && sc.getCheckpointDir.isDefined) {
      vertices.checkpoint()
    }
  }

  protected def checkpointGradientSum(delta: VertexRDD[Array[Double]]): Unit = {
    val sc = delta.sparkContext
    if (innerIter % checkpointInterval == 0 && sc.getCheckpointDir.isDefined) {
      delta.checkpoint()
    }
  }


  protected def unpersistVertices(): Unit = {
    if (previousVertices != null) previousVertices.unpersist(blocking = false)
    if (gradient != null) gradient.unpersist(blocking = false)
    if (margin != null) margin.unpersist(blocking = false)
  }
}

class LogisticRegressionSGD(
  dataSet_ : Graph[VD, ED],
  stepSize_ : Double,
  regParam_ : Double,
  useAdaGrad_ : Boolean,
  storageLevel_ : StorageLevel) extends LogisticRegression(dataSet_, stepSize_, regParam_, useAdaGrad_, storageLevel_) {
  def this(
    input: RDD[(VertexId, LabeledPoint)],
    stepSize: Double = 1e-4,
    regParam: Double = 0.0,
    useAdaGrad: Boolean = false,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK) {
    this(initializeDataSet(input, storageLevel), stepSize, regParam, useAdaGrad, storageLevel)
  }

  // SGD
  @transient private var multiplier: VertexRDD[Double] = null

  override protected[ml] def forward(iter: Int): VertexRDD[VD] = {
    dataSet.aggregateMessages[Double](ctx => {
      // val sampleId = ctx.dstId
      // val featureId = ctx.srcId
      val x = ctx.attr
      val w = ctx.srcAttr
      val z = -w * x
      assert(!z.isNaN)
      ctx.sendToDst(z)
    }, _ + _, TripletFields.Src).setName(s"margin-$iter").persist(storageLevel)
  }

  override protected[ml] def backward(q: VertexRDD[VD], iter: Int): VertexRDD[Double] = {
    multiplier = dataSet.vertices.leftJoin(q) { (_, y, margin) =>
      margin match {
        case Some(z) =>
          (1.0 / (1.0 + exp(z))) - y
        case _ => 0.0
      }
    }
    multiplier.setName(s"multiplier-$iter").persist(storageLevel)
    GraphImpl.fromExistingRDDs(multiplier, dataSet.edges).aggregateMessages[Double](ctx => {
      // val sampleId = ctx.dstId
      // val featureId = ctx.srcId
      val x = ctx.attr
      val m = ctx.dstAttr
      val grad = x * m
      ctx.sendToSrc(grad)
    }, _ + _, TripletFields.Dst).mapValues { gradient =>
      gradient / numSamples
    }.setName(s"gradient-$iter").persist(storageLevel)
  }

  override protected[ml] def loss(q: VertexRDD[VD]): Double = {
    dataSet.vertices.leftJoin(q) { case (_, y, margin) =>
      margin match {
        case Some(z) =>
          if (y > 0.0) {
            Utils.log1pExp(z)
          } else {
            Utils.log1pExp(z) - z
          }
        case _ => 0.0
      }
    }.map(_._2).reduce(_ + _) / numSamples
  }

  override protected def unpersistVertices(): Unit = {
    if (multiplier != null) multiplier.unpersist(blocking = false)
    if (margin != null) margin.unpersist(blocking = false)
    super.unpersistVertices()
  }
}

// Modified Iterative Scaling, the paper:
// A comparison of numerical optimizers for logistic regression
// http://research.microsoft.com/en-us/um/people/minka/papers/logreg/minka-logreg.pdf
class LogisticRegressionMIS(
  dataSet_ : Graph[VD, ED],
  stepSize_ : Double,
  regParam_ : Double,
  useAdaGrad_ : Boolean,
  storageLevel_ : StorageLevel) extends LogisticRegression(dataSet_, stepSize_, regParam_, useAdaGrad_, storageLevel_) {
  def this(
    input: RDD[(VertexId, LabeledPoint)],
    stepSize: Double = 1e-4,
    regParam: Double = 0.0,
    useAdaGrad: Boolean = false,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK) {
    this(initializeDataSet(input, storageLevel), stepSize, regParam, useAdaGrad, storageLevel)
  }

  @transient private var qWithLabel: VertexRDD[Double] = null
  private var epsilon = 1e-4

  def setEpsilon(eps: Double): this.type = {
    epsilon = eps
    this
  }

  override protected def backward(z: VertexRDD[VD], iter: Int): VertexRDD[Double] = {
    val q = z.mapValues { z =>
      val q = 1.0 / (1.0 + exp(z))
      // if (q.isInfinite || q.isNaN || q == 0.0) println(z)
      assert(q != 0.0)
      q
    }
    qWithLabel = dataSet.vertices.leftJoin(q) { (_, label, qv) =>
      qv.map(q => if (label > 0) q else -q).getOrElse(0)
    }
    qWithLabel.setName(s"qWithLabel-$iter").persist(storageLevel)
    GraphImpl.fromExistingRDDs(qWithLabel, dataSet.edges).aggregateMessages[Array[Double]](ctx => {
      // val sampleId = ctx.dstId
      // val featureId = ctx.srcId
      val x = ctx.attr
      val qs = ctx.dstAttr
      val q = qs * x
      assert(q != 0.0)
      val mu = if (q > 0.0) {
        Array(q, 0.0)
      } else {
        Array(0.0, -q)
      }
      ctx.sendToSrc(mu)
    }, (a, b) => Array(a(0) + b(0), a(1) + b(1)), TripletFields.Dst).mapValues { mu =>
      // TODO: 0.0 right?
      val grad = if (epsilon == 0.0) {
        if (mu.min == 0.0) 0.0 else math.log(mu(0) / mu(1))
      } else {
        math.log((mu(0) + epsilon) / (mu(1) + epsilon))
      }
      -grad
    }.setName(s"gradient-$iter").persist(storageLevel)
  }

  override protected[ml] def forward(iter: Int): VertexRDD[VD] = {
    dataSet.aggregateMessages[Double](ctx => {
      // val sampleId = ctx.dstId
      // val featureId = ctx.srcId
      val x = ctx.attr
      val w = ctx.srcAttr
      val y = ctx.dstAttr
      val z = y * w * x
      assert(!z.isNaN)
      ctx.sendToDst(z)
    }, _ + _, TripletFields.All).setName(s"q-$iter").persist(storageLevel)
  }

  override protected[ml] def loss(q: VertexRDD[VD]): Double = {
    dataSet.vertices.leftJoin(q) { case (_, y, margin) =>
      margin match {
        case Some(z) =>
          if (y > 0.0) {
            Utils.log1pExp(-z)
          } else {
            Utils.log1pExp(z) - z
          }
        case _ => 0.0
      }
    }.map(_._2).reduce(_ + _) / numSamples
  }

  override protected def unpersistVertices(): Unit = {
    if (qWithLabel != null) qWithLabel.unpersist(blocking = false)
    super.unpersistVertices()
  }

  override protected def thisIterStepSize(iter: Int): Double = {
    if (useAdaGrad) {
      stepSize * min(iter / 11.0, 1.0)
    } else {
      stepSize / sqrt(10 + iter)
    }
  }
}

object LogisticRegression {
  private[ml] type ED = Double
  private[ml] type VD = Double

  /**
   * :: Experimental ::
   * SGD training
   * @param input training data, with {0,1} label (binary classification)
   * @param numIterations maximum number of iterations
   * @param stepSize  learning step size, recommend to be 0.1 - 1.0
   * @param regParam  L1 Regularization
   * @param useAdaGrad  adaptive step size, recommend to be True
   * @param storageLevel recommendation configuration: MEMORY_AND_DISK for small/middle-scale training data,
   *                     and DISK_ONLY for super-large-scale data
   */
  @Experimental
  def trainSGD(
    input: RDD[(Long, LabeledPoint)],
    numIterations: Int,
    stepSize: Double,
    regParam: Double,
    useAdaGrad: Boolean = false,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK): GeneralizedLinearModel = {
    val data = input.map { case (id, LabeledPoint(label, features)) =>
      assert(id >= 0.0, s"sampleId $id less than 0")
      val newLabel = if (label > 0.0) 1.0 else 0.0
      (id, LabeledPoint(newLabel, features))
    }
    val lr = new LogisticRegressionSGD(data, stepSize, regParam, useAdaGrad, storageLevel)
    lr.run(numIterations)
    lr.saveModel()
  }

  /**
   * Modified Iterative Scaling
   * The referenced paper:
   * A comparison of numerical optimizers for logistic regression
   * http://research.microsoft.com/en-us/um/people/minka/papers/logreg/minka-logreg.pdf
   * @param input training data, feature value must >= 0, label is either 0 or 1 (binary classification)
   * @param numIterations maximum number of iterations
   * @param stepSize  step size, recommend to be in value range 0.1 - 1.0
   * @param regParam  L1 Regularization
   * @param epsilon   smoothing parameter, 1e-4 - 1e-6
   * @param useAdaGrad  adaptive step size, recommend to be true
   * @param storageLevel recommendation configuration: MEMORY_AND_DISK for small/middle-scale training data,
   *                     and DISK_ONLY for super-large-scale data
   */
  def trainMIS(
    input: RDD[(Long, LabeledPoint)],
    numIterations: Int,
    stepSize: Double,
    regParam: Double,
    epsilon: Double = 1e-3,
    useAdaGrad: Boolean = false,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK): GeneralizedLinearModel = {
    val data = input.map { case (id, LabeledPoint(label, features)) =>
      assert(id >= 0.0, s"sampleId $id less than 0")
      val newLabel = if (label > 0.0) 1.0 else -1.0
      features.activeValuesIterator.foreach(t => assert(t >= 0.0, s"feature $t less than 0"))
      (id, LabeledPoint(newLabel, features))
    }
    val lr = new LogisticRegressionMIS(data, stepSize, regParam, useAdaGrad, storageLevel)
    lr.setEpsilon(epsilon).run(numIterations)
    lr.saveModel()
  }

  private[ml] def initializeDataSet(
    input: RDD[(VertexId, LabeledPoint)],
    storageLevel: StorageLevel): Graph[VD, ED] = {

    val vertices = input.map { case (sampleId, labelPoint) =>
      val newId = newSampleId(sampleId)
      (newId, labelPoint.label)
    }.persist(storageLevel)

    val edges = input.flatMap { case (sampleId, labelPoint) =>
      val newId = newSampleId(sampleId)
      labelPoint.features.activeIterator.filter(_._2 != 0.0).map { case (index, value) =>
        Edge(index, newId, value)
      }
    }.persist(storageLevel)

    val dataSet = GraphImpl(vertices, edges, 0D, storageLevel, storageLevel)
    val newDataSet = DBHPartitioner.partitionByDBH(dataSet, storageLevel)
    newDataSet.persist(storageLevel)
    newDataSet.vertices.count()
    newDataSet.edges.count()
    dataSet.edges.unpersist(blocking = false)
    dataSet.vertices.unpersist(blocking = false)
    edges.unpersist(blocking = false)
    vertices.unpersist(blocking = false)
    newDataSet
  }

  private def newSampleId(id: Long): VertexId = {
    -(id + 1L)
  }
}
