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

import com.github.cloudml.zen.ml.DBHPartitioner
import com.github.cloudml.zen.ml.util.SparkUtils._
import com.github.cloudml.zen.ml.util.Utils
import org.apache.spark.annotation.Experimental
import org.apache.spark.graphx._
import org.apache.spark.graphx.impl.{EdgeRDDImpl, GraphImpl}
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.linalg.{DenseVector => SDV}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{HashPartitioner, Logging}

import scala.math._
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
  @transient protected var delta: VertexRDD[(Double, Double)] = null
  @transient protected var deltaSum: VertexRDD[Double] = null
  @transient protected var gradient: VertexRDD[Double] = null
  @transient protected var vertices = dataSet.vertices
  @transient protected var previousVertices = vertices
  @transient protected val edges = dataSet.edges.asInstanceOf[EdgeRDDImpl[ED, _]]
    .mapEdgePartitions((pid, part) => part.withoutVertexAttributes[VD]).setName("edges")

  val numFeatures: Long = features.count()
  val numSamples: Long = samples.count()

  if (edges.getStorageLevel == StorageLevel.NONE) {
    edges.persist(storageLevel)
  }
  edges.count()

  if (vertices.getStorageLevel == StorageLevel.NONE) {
    vertices.persist(storageLevel)
  }
  vertices.count()

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
      gradient = updateDeltaSum(gradient, iter)

      val tis = thisIterStepSize(iter)
      vertices = updateWeight(gradient, iter, tis, stepSize / sqrt(iter))
      checkpoint()
      vertices.count()
      dataSet = GraphImpl(vertices, edges)
      val elapsedSeconds = (System.nanoTime() - startedAt) / 1e9
      logInfo(s"train (Iteration $iter/$iterations) loss:              ${loss(margin)}")
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

  def saveModel(): LogisticRegressionModel = {
    val numFeatures = features.map(_._1).max().toInt + 1
    val featureData = new Array[Double](numFeatures)
    features.toLocalIterator.foreach { case (index, value) =>
      featureData(index.toInt) = value
    }
    new LogisticRegressionModel(new SDV(featureData), 0.0, numFeatures, 2)
  }

  protected def updateDeltaSum(gradient: VertexRDD[Double], iter: Int): VertexRDD[Double] = {
    if (gradient.getStorageLevel == StorageLevel.NONE) {
      gradient.setName(s"gradient-$iter").persist(storageLevel)
    }
    if (useAdaGrad) {
      // delta = adaGrad(deltaSum, gradient, 1.0, 1e-2, 1 - 1e-2)
      delta = adaGrad(deltaSum, gradient, 1.0, 1e-4, 1)
      delta.setName(s"delta-$iter").persist(storageLevel).count()

      gradient.unpersist(blocking = false)
      val newGradient = delta.mapValues(_._2).setName(s"gradient-$iter").persist(storageLevel)
      newGradient.count()

      if (deltaSum != null) deltaSum.unpersist(blocking = false)
      deltaSum = delta.mapValues(_._1).setName(s"deltaSum-$iter").persist(storageLevel)
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
    deltaSum: VertexRDD[Double],
    gradient: VertexRDD[Double],
    gamma: Double,
    epsilon: Double,
    rho: Double): VertexRDD[(Double, Double)] = {
    val delta = if (deltaSum == null) {
      gradient.mapValues(t => 0.0)
    }
    else {
      deltaSum
    }
    delta.innerJoin(gradient) { (_, gradSum, grad) =>
      val newGradSum = gradSum * rho + pow(grad, 2)
      val newGrad = grad * gamma / (epsilon + sqrt(newGradSum))
      (newGradSum, newGrad)
    }
  }

  protected def checkpoint(): Unit = {
    val sc = vertices.sparkContext
    if (innerIter % checkpointInterval == 0 && sc.getCheckpointDir.isDefined) {
      vertices.checkpoint()
      edges.checkpoint()
      if (deltaSum != null) deltaSum.checkpoint()
    }
  }

  protected def unpersistVertices(): Unit = {
    if (previousVertices != null) previousVertices.unpersist(blocking = false)
    if (gradient != null) gradient.unpersist(blocking = false)
    if (delta != null) delta.unpersist(blocking = false)
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
    GraphImpl(multiplier, dataSet.edges).aggregateMessages[Double](ctx => {
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

  @transient private var qWithLabel: VertexRDD[(Double, Double)] = null
  private var epsilon = 1e-4

  def setEpsilon(eps: Double): this.type = {
    epsilon = eps
    this
  }

  override protected def backward(q: VertexRDD[VD], iter: Int): VertexRDD[Double] = {
    qWithLabel = dataSet.vertices.leftJoin(q.mapValues { z =>
      val q = 1.0 / (1.0 + exp(z))
      // if (q.isInfinite || q.isNaN || q == 0.0) println(z)
      assert(q != 0.0)
      q
    }) { (_, label, qv) => (label, qv.getOrElse(0.0)) }
    qWithLabel.setName(s"qWithLabel-$iter").persist(storageLevel)
    GraphImpl(qWithLabel, dataSet.edges).aggregateMessages[Array[Double]](ctx => {
      // val sampleId = ctx.dstId
      // val featureId = ctx.srcId
      val x = ctx.attr
      val y = ctx.dstAttr._1
      val q = ctx.dstAttr._2 * abs(x)
      assert(q != 0.0)
      val mu = if (signum(x * y) > 0.0) {
        Array(q, 0.0)
      } else {
        Array(0.0, q)
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
   * 使用 SGD训练
   * @param input 训练数据 训练数据,label为{0,1} (binary classification)
   * @param numIterations 最大迭代次数
   * @param stepSize  学习步长 推荐 0.1 - 1.0
   * @param regParam  L1 Regularization
   * @param useAdaGrad  自适应步长 推荐设置为true
   * @param storageLevel 缓存级别  中小型训练集推荐设置为MEMORY_AND_DISK,大型数据集推荐设置为DISK_ONLY
   */
  @Experimental
  def trainSGD(
    input: RDD[LabeledPoint],
    numIterations: Int,
    stepSize: Double,
    regParam: Double,
    useAdaGrad: Boolean = false,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK): LogisticRegressionModel = {
    val data = input.zipWithIndex().map { case (LabeledPoint(label, features), id) =>
      val newLabel = if (label > 0.0) 1.0 else 0.0
      (id, LabeledPoint(newLabel, features))
    }.persist(storageLevel)
    val lr = new LogisticRegressionSGD(data, stepSize, regParam, useAdaGrad, storageLevel)
    data.unpersist()
    lr.run(numIterations)
    lr.saveModel
  }

  /**
   * 使用 Modified Iterative Scaling 训练 相关论文
   * A comparison of numerical optimizers for logistic regression
   * http://research.microsoft.com/en-us/um/people/minka/papers/logreg/minka-logreg.pdf
   * @param input 训练数据 每个features的值必须大于等于0,label为{0,1} (binary classification)
   * @param numIterations 最大迭代次数
   * @param stepSize  学习步长 推荐 0.1 - 1.0
   * @param regParam  L1 Regularization
   * @param epsilon   平滑参数 推荐 1e-4 - 1e-6
   * @param useAdaGrad  自适应步长 推荐设置为true
   * @param storageLevel 缓存级别  中小型训练集推荐设置为MEMORY_AND_DISK,大型数据集推荐设置为DISK_ONLY
   */
  def trainMIS(
    input: RDD[LabeledPoint],
    numIterations: Int,
    stepSize: Double,
    regParam: Double,
    epsilon: Double = 1e-3,
    useAdaGrad: Boolean = false,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK): LogisticRegressionModel = {
    val data = input.zipWithIndex().map { case (LabeledPoint(label, features), id) =>
      val newLabel = if (label > 0.0) 1.0 else -1.0
      features.activeValuesIterator.foreach(t => assert(t >= 0.0, s"feature $t less than 0"))
      (id, LabeledPoint(newLabel, features))
    }.persist(storageLevel)
    val lr = new LogisticRegressionMIS(data, stepSize, regParam, useAdaGrad, storageLevel)
    data.unpersist()
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

    // degree-based hashing
    val degrees = edges.flatMap(t => Seq((t.dstId, 1), (t.srcId, 1))).
      reduceByKey(_ + _).persist(storageLevel)
    val dataSet = GraphImpl(degrees, edges, 0, storageLevel, storageLevel)
    dataSet.persist(storageLevel)
    val numPartitions = edges.partitions.size
    val partitionStrategy = new DBHPartitioner(numPartitions)
    val newEdges = dataSet.triplets.mapPartitions { itr =>
      itr.map { e =>
        (partitionStrategy.getPartition(e), Edge(e.srcId, e.dstId, e.attr))
      }
    }.partitionBy(new HashPartitioner(numPartitions)).map(_._2)
    // end degree-based hashing

    // dataSet = dataSet.partitionBy(PartitionStrategy.EdgePartition2D)
    val newDataSet = GraphImpl(vertices, newEdges, 0.0, storageLevel, storageLevel)
    newDataSet.persist(storageLevel)
    newDataSet.vertices.count()
    newDataSet.edges.count()
    degrees.unpersist(blocking = false)
    dataSet.unpersist(blocking = false)
    edges.unpersist(blocking = false)
    vertices.unpersist(blocking = false)
    newDataSet
  }

  private def newSampleId(id: Long): VertexId = {
    -(id + 1L)
  }
}
