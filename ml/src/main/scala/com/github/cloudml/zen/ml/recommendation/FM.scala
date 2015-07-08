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

import java.util.{Random => JavaRandom}

import com.github.cloudml.zen.ml.DBHPartitioner
import com.github.cloudml.zen.ml.recommendation.FM._
import com.github.cloudml.zen.ml.util.SparkUtils._
import com.github.cloudml.zen.ml.util.{XORShiftRandom, Utils}
import org.apache.spark.{SparkContext, Logging}
import org.apache.spark.graphx._
import org.apache.spark.graphx.impl.{EdgeRDDImpl, GraphImpl}
import org.apache.spark.mllib.linalg.{Vector => SV}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.math._

/**
 * Factorization Machines 公式定义:
 * \breve{y}(x) := w_{0} + \sum_{j=1}^{n}w_{j}x_{j} + \sum_{i=1}^{n}\sum_{j=i+1}^{n}<v_{i},v_{j}> x_{i}x_{j}
 * := w_{0} + \sum_{j=1}^{n}w_{j}x_{j} +\frac{1}{2}\sum_{f=1}^{k}((\sum_{i=1}^{n}v_{i,f}x_{i})^{2}
 * - \sum_{i=1}^{n}v_{i,j}^{2}x_{i}^{2})
 * 其中<v_{i},v_{j}> :=\sum_{f=1}^{k}v_{i,j}*v_{j,f}
 */
private[ml] abstract class FM extends Serializable with Logging {

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
  @transient protected var innerIter = 1

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

  def rank: Int

  def useAdaGrad: Boolean

  def halfLife: Int = 40

  def epsilon: Double = 1e-6

  def storageLevel: StorageLevel

  def miniBatchFraction: Double

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
      val startedAt = System.nanoTime()
      val previousVertices = vertices
      val margin = forward(iter)
      var (thisNumSamples, rmse, gradient) = backward(margin, iter)
      gradient = updateGradientSum(gradient, iter)
      vertices = updateWeight(gradient, iter)
      checkpointVertices()
      vertices.count()
      dataSet = GraphImpl.fromExistingRDDs(vertices, edges)
      logInfo(s"(Iteration $iter/$iterations) RMSE:                     $rmse")
      val elapsedSeconds = (System.nanoTime() - startedAt) / 1e9
      logInfo(s"End  train (Iteration $iter/$iterations) takes:         $elapsedSeconds")

      previousVertices.unpersist(blocking = false)
      margin.unpersist(blocking = false)
      multi.unpersist(blocking = false)
      gradient._2.unpersist(blocking = false)
      innerIter += 1
    }
  }

  def saveModel(): FMModel = {
    new FMModel(rank, intercept, false, features)
  }

  protected[ml] def forward(iter: Int): VertexRDD[Array[Double]] = {
    val random: JavaRandom = new XORShiftRandom
    random.setSeed(17425170 - iter - innerIter)
    val seed = random.nextLong()
    val mod = mask
    dataSet.aggregateMessages[Array[Double]](ctx => {
      val sampleId = ctx.dstId
      // val featureId = ctx.srcId
      if (mod == 1 || isSampled(random, seed, sampleId, iter, mod)) {
        val result = forwardInterval(rank, ctx.attr, ctx.srcAttr)
        ctx.sendToDst(result)
      }
    }, reduceInterval, TripletFields.Src).setName(s"margin-$iter").persist(storageLevel)
  }

  protected def predict(arr: Array[Double]): Double

  protected def multiplier(q: VertexRDD[VD], iter: Int): (Long, Double, VertexRDD[VD])

  protected def backward(q: VertexRDD[VD], iter: Int): (Long, Double, (Double, VertexRDD[Array[Double]])) = {
    val (thisNumSamples, costSum, thisMulti) = multiplier(q, iter)
    multi = thisMulti
    val random: JavaRandom = new XORShiftRandom()
    random.setSeed(17425170 - iter - innerIter)
    val seed = random.nextLong()
    val mod = mask
    val gradW0 = multi.map(_._2.last).sum() / thisNumSamples
    val gradient = GraphImpl.fromExistingRDDs(multi, edges).aggregateMessages[Array[Double]](ctx => {
      val sampleId = ctx.dstId
      // val featureId = ctx.srcId
      if (mod == 1 || isSampled(random, seed, sampleId, iter, mod)) {
        val x = ctx.attr
        val Array(sumM, multi) = ctx.dstAttr
        val factors = ctx.srcAttr
        val m = backwardInterval(rank, x, sumM, multi, factors)
        ctx.sendToSrc(m) // send the multi directly
      }
    }, reduceInterval, TripletFields.All).mapValues { gradients =>
      gradients.map(_ / thisNumSamples)
    }
    gradient.setName(s"gradient-$iter").persist(storageLevel)
    (thisNumSamples, costSum, (gradW0, gradient))
  }

  // Updater for L2 regularized problems
  protected def updateWeight(delta: (Double, VertexRDD[Array[Double]]), iter: Int): VertexRDD[VD] = {
    val (biasGrad, gradient) = delta
    val wStepSize = if (useAdaGrad) stepSize else stepSize / sqrt(iter)
    val l2StepSize = stepSize / sqrt(iter)
    val (regB, regW, regV) = l2
    bias -= wStepSize * biasGrad + l2StepSize * regB * bias
    dataSet.vertices.leftJoin(gradient) { (_, attr, gradient) =>
      gradient match {
        case Some(grad) =>
          val weight = attr
          weight(0) -= wStepSize * grad(0) + l2StepSize * regW * weight(0)
          var i = 1
          while (i <= rank) {
            weight(i) -= wStepSize * grad(i) + l2StepSize * regV * weight(i)
            i += 1
          }
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
      // val (newW0Grad, newW0Sum, delta) = esgd(gradientSum, gradient, epsilon, iter)
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

class FMClassification(
  @transient _dataSet: Graph[VD, ED],
  val stepSize: Double,
  val l2: (Double, Double, Double),
  val rank: Int,
  val useAdaGrad: Boolean,
  val miniBatchFraction: Double,
  val storageLevel: StorageLevel) extends FM {

  def this(
    input: RDD[(VertexId, LabeledPoint)],
    stepSize: Double = 1e-2,
    l2Reg: (Double, Double, Double) = (1e-3, 1e-3, 1e-3),
    rank: Int = 20,
    useAdaGrad: Boolean = true,
    miniBatchFraction: Double = 1.0,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK) {
    this(initializeDataSet(input, rank, storageLevel), stepSize, l2Reg, rank,
      useAdaGrad, miniBatchFraction, storageLevel)
  }

  setDataSet(_dataSet)

  // assert(rank > 1, s"rank $rank less than 2")

  override protected def predict(arr: Array[Double]): Double = {
    val result = predictInterval(rank, bias, arr)
    1.0 / (1.0 + math.exp(-result))
  }

  override def saveModel(): FMModel = {
    new FMModel(rank, intercept, true, features)
  }

  override protected def multiplier(q: VertexRDD[VD], iter: Int): (Long, Double, VertexRDD[VD]) = {
    val random: JavaRandom = new XORShiftRandom()
    random.setSeed(17425170 - iter - innerIter)
    val seed = random.nextLong()
    val mod = mask
    val multi = dataSet.vertices.leftJoin(q) { (vid, data, deg) =>
      deg match {
        case Some(m) =>
          val y = data.head
          val diff = predict(m) - y
          val z = predictInterval(rank, bias, m)
          val loss = if (y > 0.0) Utils.log1pExp(-z) else Utils.log1pExp(z)
          (Array(sumInterval(rank, m), diff), loss)
        case _ => (data, 0.0)
      }
    }
    multi.setName(s"multiplier-$iter").persist(storageLevel)
    val Array(numSamples, costSum) = multi.filter { case (id, _) =>
      isSampleId(id) && (mod == 1 || isSampled(random, seed, id, iter, mod))
    }.map { case (_, (arr, loss)) =>
      Array(1D, loss)
    }.reduce(reduceInterval)
    (numSamples.toLong, costSum / numSamples, multi.mapValues(_._1))
  }

}

class FMRegression(
  @transient _dataSet: Graph[VD, ED],
  val stepSize: Double,
  val l2: (Double, Double, Double),
  val rank: Int,
  val useAdaGrad: Boolean,
  val miniBatchFraction: Double,
  val storageLevel: StorageLevel) extends FM {

  def this(
    input: RDD[(VertexId, LabeledPoint)],
    stepSize: Double = 1e-2,
    l2Reg: (Double, Double, Double) = (1e-3, 1e-3, 1e-3),
    rank: Int = 20,
    useAdaGrad: Boolean = true,
    miniBatchFraction: Double = 1.0,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK) {
    this(initializeDataSet(input, rank, storageLevel), stepSize, l2Reg, rank,
      useAdaGrad, miniBatchFraction, storageLevel)
  }

  setDataSet(_dataSet)

  // assert(rank > 1, s"rank $rank less than 2")

  // val max = samples.map(_._2.head).max
  // val min = samples.map(_._2.head).min

  override protected def predict(arr: Array[Double]): Double = {
    var result = predictInterval(rank, bias, arr)
    // result = Math.max(result, min)
    // result = Math.min(result, max)
    result
  }

  override protected def multiplier(q: VertexRDD[VD], iter: Int): (Long, Double, VertexRDD[VD]) = {
    val random: JavaRandom = new XORShiftRandom()
    random.setSeed(17425170 - iter - innerIter)
    val seed = random.nextLong()
    val mod = mask
    val multi = dataSet.vertices.leftJoin(q) { (vid, data, deg) =>
      deg match {
        case Some(m) =>
          val y = data.head
          val diff = predict(m) - y
          Array(sumInterval(rank, m), diff * 2.0)
        case _ => data
      }
    }
    multi.setName(s"multiplier-$iter").persist(storageLevel)
    val Array(numSamples, costSum) = multi.filter { case (id, _) =>
      isSampleId(id) && (mod == 1 || isSampled(random, seed, id, iter, mod))
    }.map { case (_, arr) =>
      Array(1.0, pow(arr.last / 2.0, 2.0))
    }.reduce(reduceInterval)
    (numSamples.toLong, sqrt(costSum / numSamples), multi)
  }
}

object FM {
  private[ml] type ED = Double
  private[ml] type VD = Array[Double]

  /**
   * FM clustering
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
  def trainClassification(
    input: RDD[(Long, LabeledPoint)],
    numIterations: Int,
    stepSize: Double,
    l2: (Double, Double, Double),
    rank: Int,
    useAdaGrad: Boolean = true,
    miniBatchFraction: Double = 1.0,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK): FMModel = {
    val data = input.map { case (id, LabeledPoint(label, features)) =>
      assert(id >= 0.0, s"sampleId $id less than 0")
      val newLabel = if (label > 0.0) 1.0 else 0.0
      (id, LabeledPoint(newLabel, features))
    }
    val lfm = new FMClassification(data, stepSize, l2, rank, useAdaGrad, miniBatchFraction, storageLevel)
    lfm.run(numIterations)
    val model = lfm.saveModel()
    model
  }

  /**
   * FM regression
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
    l2: (Double, Double, Double),
    rank: Int,
    useAdaGrad: Boolean = true,
    miniBatchFraction: Double = 1.0,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK): FMModel = {
    val data = input.map { case (id, labeledPoint) =>
      assert(id >= 0.0, s"sampleId $id less than 0")
      (id, labeledPoint)
    }
    val lfm = new FMRegression(data, stepSize, l2, rank, useAdaGrad, miniBatchFraction, storageLevel)
    lfm.run(numIterations)
    val model = lfm.saveModel()
    model
  }

  private[ml] def initializeDataSet(
    input: RDD[(VertexId, LabeledPoint)],
    rank: Int,
    storageLevel: StorageLevel): Graph[VD, ED] = {
    val edges = input.flatMap { case (sampleId, labelPoint) =>
      // sample id
      val newId = newSampleId(sampleId)
      val features = labelPoint.features
      features.activeIterator.filter(_._2 != 0.0).map { case (featureId, value) =>
        Edge(featureId, newId, value)
      }
    }.persist(storageLevel)
    edges.count()

    val vertices = (input.map { case (sampleId, labelPoint) =>
      val newId = newSampleId(sampleId)
      // label point
      val label = Array(labelPoint.label)
      (newId, label)
    } ++ edges.map(_.srcId).distinct().map { featureId =>
      // parameter point
      val parms = Array.fill(rank + 1) {
        Utils.random.nextGaussian() * 1e-2
      }
      parms(0) = 0.0
      (featureId, parms)
    }).repartition(input.partitions.length)
    vertices.persist(storageLevel)
    vertices.count()

    val dataSet = GraphImpl(vertices, edges, null.asInstanceOf[VD], storageLevel, storageLevel)
    val newDataSet = DBHPartitioner.partitionByDBH(dataSet, storageLevel)
    edges.unpersist()
    vertices.unpersist()
    newDataSet
  }

  @inline private[ml] def newSampleId(id: Long): VertexId = {
    -(id + 1L)
  }

  @inline private[ml] def isSampleId(id: Long): Boolean = {
    id < 0
  }

  @inline private[ml] def isSampled(
    random: JavaRandom,
    seed: Long,
    sampleId: Long,
    iter: Int,
    mod: Int): Boolean = {
    random.setSeed(seed * sampleId)
    random.nextInt(mod) == iter % mod
  }

  /**
   * arr[0] = \sum_{j=1}^{n}w_{j}x_{i}
   * arr[f] = \sum_{i=1}^{n}v_{i,f}x_{i} f属于 [1,rank]
   * arr[k] = \sum_{i=1}^{n} v_{i,k}^{2}x_{i}^{2} k属于 (rank,rank * 2 + 1]
   */
  private[ml] def predictInterval(rank: Int, bias: Double, arr: VD): ED = {
    var sum = 0.0
    var i = 1
    while (i <= rank) {
      sum += pow(arr(i), 2) - arr(rank + i)
      i += 1
    }
    bias + arr(0) + 0.5 * sum
  }

  private[ml] def reduceInterval(a: VD, b: VD): VD = {
    var i = 0
    while (i < a.length) {
      a(i) += b(i)
      i += 1
    }
    a
  }

  /**
   * arr[0] = \sum_{j=1}^{n}w_{j}x_{i}
   * arr[f] = \sum_{i=1}^{n}v_{i,f}x_{i}, f belongs to  [1,rank]
   * arr[k] = \sum_{i=1}^{n} v_{i,k}^{2}x_{i}^{2}, k belongs to  (rank,rank * 2 + 1]
   */
  private[ml] def forwardInterval(rank: Int, x: ED, w: VD): VD = {
    val arr = new Array[Double](rank * 2 + 1)
    arr(0) = x * w(0)
    var i = 1
    while (i <= rank) {
      arr(i) = x * w(i)
      arr(rank + i) = pow(x, 2) * pow(w(i), 2)
      i += 1
    }
    arr
  }

  /**
   * sumM =  \sum_{j=1}^{n}v_{j,f}x_{j}
   */
  private[ml] def backwardInterval(
    rank: Int,
    x: ED,
    sumM: ED,
    multi: ED,
    factors: VD): VD = {
    val m = new Array[Double](rank + 1)
    m(0) = x * multi
    var i = 1
    while (i <= rank) {
      val grad = sumM * x - factors(i) * pow(x, 2)
      m(i) = grad * multi
      i += 1
    }
    m
  }

  private[ml] def sumInterval(rank: Int, arr: Array[Double]): Double = {
    var result = 0.0
    var i = 1
    while (i <= rank) {
      result += arr(i)
      i += 1
    }
    result
  }
}
