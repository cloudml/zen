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

package com.github.cloudml.zen.ml.recommendation

import com.github.cloudml.zen.ml.DBHPartitioner
import com.github.cloudml.zen.ml.recommendation.BSFM._
import com.github.cloudml.zen.ml.util.SparkUtils._
import com.github.cloudml.zen.ml.util.Utils
import org.apache.commons.math3.primes.Primes
import org.apache.spark.{SparkContext, Logging}
import org.apache.spark.graphx._
import org.apache.spark.graphx.impl.{EdgeRDDImpl, GraphImpl}
import org.apache.spark.mllib.linalg.{Vector => SV}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.math._

/**
 * Block Structures Factorization Machines :
 * \hat{y}(x)= w_0 + \sum_{i=1}^{n}w_{i}x_{i} + \frac{1}{2}\sum_{f}^{k}((\sum_{i}^{n}{x_{i}v_{i,f}})^2 -
 * \sum_{l=1}^{|B|}(\sum_{i \epsilon B_{l}}x_{i}v_{i,f})^2)
 * the derivative: \frac{\partial \hat{y}(x|\Theta )}{\partial\theta}
 * if \theta is w_0, the derivative: 1
 * if \theta is w_i, the derivative: x_i
 * if \theta is v_{i,f}, the derivative: x_{i}(\sum_{i=1}^{n}x_{i}v_{i,f} - \sum_{i\epsilon B_{l}}x_{i}v_{i,f})
 */
private[ml] abstract class BSFM extends Serializable with Logging {

  protected val checkpointInterval = 10
  protected var bias: Double = 0.0
  protected var numFeatures: Long = 0
  protected var numSamples: Long = 0

  // SGD
  @transient protected var dataSet: Graph[VD, ED] = null
  @transient protected var multi: VertexRDD[Array[Double]] = null
  @transient protected var gradientSum: (Double, VertexRDD[Array[Double]]) = null
  @transient protected var vertices: VertexRDD[VD] = null
  @transient protected var edges: EdgeRDD[ED] = null
  @transient private var innerIter = 1
  @transient private var primes = Primes.nextPrime(117)

  def setDataSet(data: Graph[VD, ED]): this.type = {
    vertices = data.vertices
    edges = data.edges.asInstanceOf[EdgeRDDImpl[ED, _]].mapEdgePartitions { (pid, part) =>
      part.withoutVertexAttributes[VD]
    }.setName("edges").persist(storageLevel)
    if (vertices.getStorageLevel == StorageLevel.NONE) {
      vertices.persist(storageLevel)
    }
    if (edges.sparkContext.getCheckpointDir.isDefined) {
      edges.checkpoint()
      edges.count()
    }
    data.edges.unpersist(blocking = false)
    dataSet = GraphImpl.fromExistingRDDs(vertices, edges)
    numFeatures = features.count()
    numSamples = samples.count()
    this
  }

  def stepSize: Double

  def l2: (Double, Double, Double)

  def views: Array[Long]

  def rank: Int

  def useAdaGrad: Boolean

  def useWeightedLambda: Boolean

  def storageLevel: StorageLevel

  def miniBatchFraction: Double

  def halfLife: Int = 40

  def epsilon: Double = 1e-6

  def intercept: Double = {
    bias
  }

  protected[ml] def mask: Int = {
    max(1 / miniBatchFraction, 1).toInt
  }

  def samples: VertexRDD[VD] = {
    dataSet.vertices.filter(t => t._1 < 0)
  }

  def features: VertexRDD[VD] = {
    dataSet.vertices.filter(t => t._1 >= 0)
  }

  // Factorization Machines
  def run(iterations: Int): Unit = {
    for (iter <- 1 to iterations) {
      logInfo(s"Start train (Iteration $iter/$iterations)")
      primes = Primes.nextPrime(primes + 1)
      val startedAt = System.nanoTime()
      val previousVertices = vertices
      val margin = forward(iter)
      var gradient = backward(margin, iter)
      gradient = updateGradientSum(gradient, iter)
      vertices = updateWeight(gradient, iter)
      checkpointVertices()
      vertices.count()
      dataSet = GraphImpl.fromExistingRDDs(vertices, edges)
      val elapsedSeconds = (System.nanoTime() - startedAt) / 1e9
      logInfo(s"Train (Iteration $iter/$iterations) RMSE:               ${loss(margin)}")
      logInfo(s"End  train (Iteration $iter/$iterations) takes:         $elapsedSeconds")

      previousVertices.unpersist(blocking = false)
      margin.unpersist(blocking = false)
      multi.unpersist(blocking = false)
      gradient._2.unpersist(blocking = false)
      innerIter += 1
    }
  }

  def saveModel(): BSFMModel = {
    new BSFMModel(rank, intercept, views, false, features.mapValues(arr => arr.slice(0, arr.length - 1)))
  }

  protected[ml] def loss(q: VertexRDD[VD]): Double = {
    val thisNumSamples = (1.0 / mask) * numSamples
    val sum = samples.join(q).map { case (_, (y, m)) =>
      val pm = predict(m)
      // if (Utils.random.nextDouble() < 0.001) println(f"$pm%1.2f : ${y(0)}")
      pow(pm - y(0), 2)
    }.reduce(_ + _)
    sqrt(sum / thisNumSamples)
  }


  protected[ml] def forward(iter: Int): VertexRDD[Array[Double]] = {
    val mod = mask
    val thisMask = iter % mod
    val thisPrimes = primes
    dataSet.aggregateMessages[Array[Double]](ctx => {
      val sampleId = ctx.dstId
      val featureId = ctx.srcId
      val viewId = featureId2viewId(featureId, views)
      if (mod == 1 || ((sampleId * thisPrimes) % mod) + thisMask == 0) {
        val result = forwardInterval(rank, views.length, viewId, ctx.attr, ctx.srcAttr)
        ctx.sendToDst(result)
      }
    }, forwardReduceInterval, TripletFields.Src).setName(s"margin-$iter").persist(storageLevel)
  }

  protected def predict(arr: Array[Double]): Double

  protected def multiplier(q: VertexRDD[VD]): VertexRDD[VD]

  protected def backward(q: VertexRDD[VD], iter: Int): (Double, VertexRDD[Array[Double]]) = {
    val thisNumSamples = (1.0 / mask) * numSamples
    multi = multiplier(q).setName(s"multiplier-$iter").persist(storageLevel)
    val gradW0 = multi.map(_._2.last).sum() / thisNumSamples
    val sampledArrayLen = rank * views.length + 2
    val gradient = GraphImpl.fromExistingRDDs(multi, edges).aggregateMessages[Array[Double]](ctx => {
      // val sampleId = ctx.dstId
      val featureId = ctx.srcId
      if (ctx.dstAttr.length == sampledArrayLen) {
        val x = ctx.attr
        val arr = ctx.dstAttr
        val viewId = featureId2viewId(featureId, views)
        val m = backwardInterval(rank, viewId, x, arr, arr.last)
        ctx.sendToSrc(m)
      }
    }, forwardReduceInterval, TripletFields.Dst).mapValues { gradients =>
      gradients.map(_ / thisNumSamples)
    }
    gradient.setName(s"gradient-$iter").persist(storageLevel)
    (gradW0, gradient)
  }

  // Updater for L2 regularized problems
  protected def updateWeight(delta: (Double, VertexRDD[Array[Double]]), iter: Int): VertexRDD[VD] = {
    val (biasGrad, gradient) = delta
    val wStepSize = if (useAdaGrad) stepSize else stepSize / sqrt(iter)
    val (regB, regW, regV) = l2
    bias -= wStepSize * (biasGrad + regB * bias)
    dataSet.vertices.leftJoin(gradient) { (_, attr, gradient) =>
      gradient match {
        case Some(grad) =>
          val weight = attr
          val wd = if (useWeightedLambda) weight.last / (numSamples + 1.0) else 1.0
          var i = 0
          while (i < rank) {
            weight(i) -= wStepSize * (grad(i) + wd * regV * weight(i))
            i += 1
          }
          weight(rank) -= wStepSize * (grad(rank) + wd * regW * weight(rank))
          weight
        case None => attr
      }
    }.setName(s"vertices-$iter").persist(storageLevel)
  }

  protected def updateGradientSum(
    gradient: (Double, VertexRDD[Array[Double]]),
    iter: Int): (Double, VertexRDD[Array[Double]]) = {
    if (useAdaGrad) {
      val rho = math.exp(-math.log(2.0) / halfLife)
      val (newW0Grad, newW0Sum, delta) = adaGrad(gradientSum, gradient, epsilon, 1.0)
      // val (newW0Grad, newW0Sum, delta) = esgd(gradientSum, gradient, 1e-4, iter)
      checkpointGradientSum(delta)
      delta.setName(s"delta-$iter").persist(storageLevel).count()

      gradient._2.unpersist(blocking = false)
      val newGradient = delta.mapValues(_._1).filter(_._2 != null).
        setName(s"gradient-$iter").persist(storageLevel)
      newGradient.count()

      if (gradientSum != null) gradientSum._2.unpersist(blocking = false)
      gradientSum = (newW0Sum, delta.mapValues(_._2).setName(s"gradientSum-$iter").persist(storageLevel))
      gradientSum._2.count()
      delta.unpersist(blocking = false)
      (newW0Grad, newGradient)
    } else {
      gradient
    }
  }

  protected def adaGrad(
    gradientSum: (Double, VertexRDD[Array[Double]]),
    gradient: (Double, VertexRDD[Array[Double]]),
    epsilon: Double,
    rho: Double): (Double, Double, VertexRDD[(Array[Double], Array[Double])]) = {
    val delta = if (gradientSum == null) {
      features.mapValues(t => t.map(x => 0.0))
    }
    else {
      gradientSum._2
    }
    val newGradSumWithoutW0 = delta.leftJoin(gradient._2) { (_, gradSum, g) =>
      g match {
        case Some(grad) =>
          val gradLen = grad.length
          val newGradSum = new Array[Double](gradLen)
          val newGrad = new Array[Double](gradLen)
          for (i <- 0 until gradLen) {
            newGradSum(i) = gradSum(i) * rho + pow(grad(i), 2)
            newGrad(i) = grad(i) / (epsilon + sqrt(newGradSum(i)))
          }
          (newGrad, newGradSum)
        case _ => (null, gradSum)
      }

    }
    val w0Sum = if (gradientSum == null) 0.0 else gradientSum._1
    val w0Grad = gradient._1

    val newW0Sum = w0Sum * rho + pow(w0Grad, 2)
    val newW0Grad = w0Grad / (epsilon + sqrt(newW0Sum))

    (newW0Grad, newW0Sum, newGradSumWithoutW0)
  }

  protected def esgd(
    gradientSum: (Double, VertexRDD[Array[Double]]),
    gradient: (Double, VertexRDD[Array[Double]]),
    epsilon: Double,
    iter: Int): (Double, Double, VertexRDD[(Array[Double], Array[Double])]) = {
    val delta = if (gradientSum == null) {
      features.mapValues(t => t.map(x => 0.0))
    }
    else {
      gradientSum._2
    }
    val newGradSumWithoutW0 = delta.leftJoin(gradient._2) { (_, gradSum, g) =>
      g match {
        case Some(grad) =>
          val gradLen = grad.length
          val newGradSum = new Array[Double](gradLen)
          val newGrad = new Array[Double](gradLen)
          for (i <- 0 until gradLen) {
            newGradSum(i) = gradSum(i) + pow(Utils.random.nextGaussian() * grad(i), 2)
            newGrad(i) = grad(i) / (epsilon + sqrt(newGradSum(i) / iter))
          }
          (newGrad, newGradSum)
        case _ => (null, gradSum)
      }

    }
    val w0Sum = if (gradientSum == null) 0.0 else gradientSum._1
    val w0Grad = gradient._1

    val newW0Sum = w0Sum + pow(Utils.random.nextGaussian() * w0Grad, 2)
    val newW0Grad = w0Grad / (epsilon + sqrt(newW0Sum / iter))

    (newW0Grad, newW0Sum, newGradSumWithoutW0)
  }

  protected def checkpointGradientSum(delta: VertexRDD[(Array[Double], Array[Double])]): Unit = {
    val sc = delta.sparkContext
    if (innerIter % checkpointInterval == 0 && sc.getCheckpointDir.isDefined) {
      delta.checkpoint()
    }
  }

  protected def checkpointVertices(): Unit = {
    val sc = vertices.sparkContext
    if (innerIter % checkpointInterval == 0 && sc.getCheckpointDir.isDefined) {
      vertices.checkpoint()
    }
  }
}

class BSFMClassification(
  @transient _dataSet: Graph[VD, ED],
  val stepSize: Double,
  val views: Array[Long],
  val l2: (Double, Double, Double),
  val rank: Int,
  val useAdaGrad: Boolean,
  val useWeightedLambda: Boolean,
  val miniBatchFraction: Double,
  val storageLevel: StorageLevel) extends BSFM {

  def this(
    input: RDD[(VertexId, LabeledPoint)],
    stepSize: Double = 1e-2,
    views: Array[Long],
    l2Reg: (Double, Double, Double) = (1e-3, 1e-3, 1e-3),
    rank: Int = 20,
    useAdaGrad: Boolean = true,
    useWeightedLambda: Boolean = true,
    miniBatchFraction: Double = 1.0,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK) {
    this(initializeDataSet(input, views, rank, storageLevel), stepSize, views, l2Reg, rank,
      useAdaGrad, useWeightedLambda, miniBatchFraction, storageLevel)
  }

  setDataSet(_dataSet)
  assert(rank > 1, s"rank $rank less than 2")

  override protected def predict(arr: Array[Double]): Double = {
    val result = predictInterval(rank, views.length, bias, arr)
    1.0 / (1.0 + math.exp(-result))
  }

  override def saveModel(): BSFMModel = {
    new BSFMModel(rank, intercept, views, true, features.mapValues(arr => arr.slice(0, arr.length - 1)))
  }

  override protected def multiplier(q: VertexRDD[VD]): VertexRDD[VD] = {
    dataSet.vertices.leftJoin(q) { (vid, data, deg) =>
      deg match {
        case Some(m) =>
          val y = data.head
          val diff = predict(m) - y
          val ret = sumInterval(rank, views.length, m)
          ret(ret.length - 1) = diff
          ret
        case _ => data
      }
    }
  }

  override protected[ml] def loss(q: VertexRDD[VD]): Double = {
    val thisNumSamples = (1.0 / mask) * numSamples
    val sum = samples.join(q).map { case (_, (y, m)) =>
      val z = predictInterval(rank, views.length, bias, m)
      if (y(0) > 0.0) Utils.log1pExp(-z) else Utils.log1pExp(z)
    }.reduce(_ + _)
    sum / thisNumSamples
  }
}

class BSFMRegression(
  @transient _dataSet: Graph[VD, ED],
  val stepSize: Double,
  val views: Array[Long],
  val l2: (Double, Double, Double),
  val rank: Int,
  val useAdaGrad: Boolean,
  val useWeightedLambda: Boolean,
  val miniBatchFraction: Double,
  val storageLevel: StorageLevel) extends BSFM {

  def this(
    input: RDD[(VertexId, LabeledPoint)],
    stepSize: Double = 1e-2,
    views: Array[Long],
    l2Reg: (Double, Double, Double) = (1e-3, 1e-3, 1e-3),
    rank: Int = 20,
    useAdaGrad: Boolean = true,
    useWeightedLambda: Boolean = true,
    miniBatchFraction: Double = 1.0,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK) {
    this(initializeDataSet(input, views, rank, storageLevel), stepSize, views, l2Reg, rank,
      useAdaGrad, useWeightedLambda, miniBatchFraction, storageLevel)
  }

  setDataSet(_dataSet)
  assert(rank > 1, s"rank $rank less than 2")

  // val max = samples.map(_._2.head).max
  // val min = samples.map(_._2.head).min

  override protected def predict(arr: Array[Double]): Double = {
    var result = predictInterval(rank, views.length, bias, arr)
    // result = Math.max(result, min)
    // result = Math.min(result, max)
    result
  }

  override protected def multiplier(q: VertexRDD[VD]): VertexRDD[VD] = {
    dataSet.vertices.leftJoin(q) { (vid, data, deg) =>
      deg match {
        case Some(m) =>
          val y = data.head
          val diff = predict(m) - y
          val ret = sumInterval(rank, views.length, m)
          ret(ret.length - 1) = 2.0 * diff
          ret
        case _ => data
      }
    }
  }
}

object BSFM {
  private[ml] type ED = Double
  private[ml] type VD = Array[Double]

  /**
   * BS-FM clustering
   * @param input train data
   * @param numIterations
   * @param stepSize  recommend 1e-2- 1e-1
   * @param views  特征视图
   * @param l2   (w_0, w_i, v_{i,f}) in L2 regularization
   * @param rank   recommend 10-20
   * @param useAdaGrad use AdaGrad to train
   * @param miniBatchFraction
   * @param storageLevel
   * @return
   */
  def trainClassification(
    input: RDD[(Long, LabeledPoint)],
    numIterations: Int,
    stepSize: Double,
    views: Array[Long],
    l2: (Double, Double, Double),
    rank: Int,
    useAdaGrad: Boolean = true,
    useWeightedLambda: Boolean = true,
    miniBatchFraction: Double = 1.0,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK): BSFMModel = {
    val data = input.map { case (id, LabeledPoint(label, features)) =>
      assert(id >= 0.0, s"sampleId $id less than 0")
      val newLabel = if (label > 0.0) 1.0 else 0.0
      (id, LabeledPoint(newLabel, features))
    }
    val lfm = new BSFMClassification(data, stepSize, views, l2, rank, useAdaGrad,
      useWeightedLambda, miniBatchFraction, storageLevel)
    lfm.run(numIterations)
    val model = lfm.saveModel()
    model
  }

  /**
   * BS-FM regression
   * @param input train data
   * @param numIterations
   * @param stepSize  recommend 1e-2- 1e-1
   * @param l2   (w_0, w_i, v_{i,f}) in L2 regularization
   * @param rank   recommend 10-20
   * @param useAdaGrad use AdaGrad to train
   * @param miniBatchFraction
   * @param storageLevel
   * @return
   */
  def trainRegression(
    input: RDD[(Long, LabeledPoint)],
    numIterations: Int,
    stepSize: Double,
    views: Array[Long],
    l2: (Double, Double, Double),
    rank: Int,
    useAdaGrad: Boolean = true,
    useWeightedLambda: Boolean = true,
    miniBatchFraction: Double = 1.0,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK): BSFMModel = {
    val data = input.map { case (id, labeledPoint) =>
      assert(id >= 0.0, s"sampleId $id less than 0")
      (id, labeledPoint)
    }
    val lfm = new BSFMRegression(data, stepSize, views, l2, rank, useAdaGrad,
      useWeightedLambda, miniBatchFraction, storageLevel)
    lfm.run(numIterations)
    val model = lfm.saveModel()
    model
  }

  private[ml] def initializeDataSet(
    input: RDD[(VertexId, LabeledPoint)],
    views: Array[Long],
    rank: Int,
    storageLevel: StorageLevel): Graph[VD, ED] = {
    val numFeatures = input.first()._2.features.size
    assert(numFeatures == views.last)

    val edges = input.flatMap { case (sampleId, labelPoint) =>
      // sample id
      val newId = newSampleId(sampleId)
      val features = labelPoint.features
      features.activeIterator.filter(_._2 != 0.0).map { case (featureId, value) =>
        Edge(featureId, newId, value)
      }
    }.persist(storageLevel)
    edges.count()

    val inDegrees = edges.map(e => (e.srcId, 1L)).reduceByKey(_ + _).map {
      case (featureId, deg) =>
        (featureId, deg)
    }
    val features = edges.map(_.srcId).distinct().map { featureId =>
      // parameter point
      val parms = Array.fill(rank + 2) {
        Utils.random.nextGaussian() * 1e-2
      }
      (featureId, parms)
    }.join(inDegrees).map { case (featureId, (parms, deg)) =>
      parms(parms.length - 1) = deg
      (featureId, parms)
    }
    val vertices = (input.map { case (sampleId, labelPoint) =>
      val newId = newSampleId(sampleId)
      // label point
      val label = Array(labelPoint.label)
      (newId, label)
    } ++ features).repartition(input.partitions.length)
    vertices.persist(storageLevel)
    vertices.count()

    val dataSet = GraphImpl(vertices, edges, null.asInstanceOf[VD], storageLevel, storageLevel)
    val newDataSet = DBHPartitioner.partitionByDBH(dataSet, storageLevel)
    edges.unpersist()
    vertices.unpersist()
    newDataSet
  }

  @inline private[ml] def featureId2viewId(featureId: Long, views: Array[Long]): Int = {
    val numFeatures = views.last
    val viewId = if (featureId >= numFeatures) {
      featureId - numFeatures
    } else {
      val viewSize = views.length
      var adj = 0
      var found = false
      while (adj < viewSize - 1 && !found) {
        if (featureId < views(adj)) {
          found = true
        } else {
          adj += 1
        }
      }
      adj
    }
    viewId.toInt
  }

  private[ml] def newSampleId(id: Long): VertexId = {
    -(id + 1L)
  }

  /**
   * arr(k + v * rank) = \sum_{i \not{\epsilon} B_l}x_{i}v_{i,f}
   * arr(viewSize * rank) = \sum_{i=1}^{n} x_{i}w{i}
   */
  private[ml] def predictInterval(rank: Int, viewSize: Int, bias: Double, arr: VD): ED = {
    val wx = arr.last
    var sum2order = 0.0
    var i = 0
    while (i < rank) {
      var allSum = 0.0
      var viewSumP2 = 0.0
      var viewId = 0
      while (viewId < viewSize) {
        viewSumP2 += pow(arr(i + viewId * rank), 2)
        allSum += arr(i + viewId * rank)
        viewId += 1
      }
      sum2order += pow(allSum, 2) - viewSumP2
      i += 1
    }
    bias + wx + 0.5 * sum2order
  }


  private[ml] def forwardReduceInterval(a: VD, b: VD): VD = {
    var i = 0
    while (i < a.length) {
      a(i) += b(i)
      i += 1
    }
    a
  }

  private[ml] def forwardInterval(rank: Int, viewSize: Int, viewId: Int, x: ED, w: VD): VD = {
    val arr = new Array[Double](rank * viewSize + 1)
    forwardInterval(rank, viewId, arr, x, w)
  }


  /**
   * arr = rank * viewSize + 1
   * when f belongs to [viewId * rank ,(viewId +1) * rank)
   * arr[f] = x_{i}v_{i,f}
   */
  private[ml] def forwardInterval(rank: Int, viewId: Int, arr: Array[Double], z: ED, w: VD): VD = {
    arr(arr.length - 1) += z * w(rank)
    var i = 0
    while (i < rank) {
      arr(i + viewId * rank) += z * w(i)
      i += 1
    }
    arr
  }

  /**
   * m = rank + 1
   * when k belongs to [0,rank)
   * arr(k) = multi \sum_{i \not{\epsilon} B_l}x_{i}v_{i,f}
   * arr(rank)= multi x
   * clustering: multi = 1/(1+ \exp(-\hat{y}(x|\Theta)) ) - y
   * regression: multi = 2(\hat{y}(x|\Theta) -y)
   */
  private[ml] def backwardInterval(
    rank: Int,
    viewId: Int,
    x: ED,
    arr: VD,
    multi: ED): VD = {
    val m = new Array[Double](rank + 1)
    var i = 0
    while (i < rank) {
      m(i) = multi * x * arr(i + viewId * rank)
      i += 1
    }
    m(rank) = x * multi
    m
  }

  /**
   * arr(k + v * rank)= \sum_{i \not{\epsilon} B_l}x_{i}v_{i,f}
   * arr(rank * viewSize) = x_l
   */
  private[ml] def sumInterval(rank: Int, viewSize: Int, arr: Array[Double]): VD = {
    val m = new Array[Double](rank)
    var i = 0
    while (i < rank) {
      var multi = 0.0
      var viewId = 0
      while (viewId < viewSize) {
        multi += arr(i + viewId * rank)
        viewId += 1
      }
      m(i) += multi
      i += 1
    }

    val ret = new Array[Double](rank * viewSize + 2)
    i = 0
    while (i < rank) {
      var viewId = 0
      while (viewId < viewSize) {
        ret(i + viewId * rank) = m(i) - arr(i + viewId * rank)
        viewId += 1
      }
      i += 1
    }
    ret(rank * viewSize) = arr(rank * viewSize)
    ret
  }
}
