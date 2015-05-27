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

import com.github.cloudml.zen.ml.recommendation.FM._
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

  def regB: Double

  def regW: Double

  def regV: Double

  def rank: Int

  def useAdaGrad: Boolean

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
      // logInfo(s"Train (Iteration $iter/$iterations) cost:               ${loss(margin)}")
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

  protected[ml] def loss(q: VertexRDD[VD]): Double = {
    val thisNumSamples = (1.0 / mask) * numSamples
    samples.join(q).map { case (_, (y, m)) =>
      val pm = predict(m)
      // if (Utils.random.nextDouble() < 0.001) println(f"$pm%1.2f : ${y(0)}")
      0.5 * pow(pm - y(0), 2)
    }.reduce(_ + _) / thisNumSamples
  }

  protected[ml] def forward(iter: Int): VertexRDD[Array[Double]] = {
    val mod = mask
    val thisMask = iter % mod
    val thisPrimes = primes
    dataSet.aggregateMessages[Array[Double]](ctx => {
      val sampleId = ctx.dstId
      // val featureId = ctx.srcId
      if (mod == 1 || ((sampleId * thisPrimes) % mod) + thisMask == 0) {
        val result = forwardInterval(rank, ctx.attr, ctx.srcAttr)
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
    val gradient = GraphImpl.fromExistingRDDs(multi, edges).aggregateMessages[Array[Double]](ctx => {
      // val sampleId = ctx.dstId
      // val featureId = ctx.srcId
      if (ctx.dstAttr.length == 2) {
        val x = ctx.attr
        val Array(sumM, multi) = ctx.dstAttr
        val factors = ctx.srcAttr
        val m = backwardInterval(rank, x, sumM, multi, factors)
        ctx.sendToSrc(m) // send the multi directly
      }
    }, forwardReduceInterval, TripletFields.All).mapValues { gradients =>
      gradients.map(_ / thisNumSamples) // / numSamples
    }
    gradient.setName(s"gradient-$iter").persist(storageLevel)
    (gradW0, gradient)
  }

  // Updater for L2 regularized problems
  protected def updateWeight(delta: (Double, VertexRDD[Array[Double]]), iter: Int): VertexRDD[VD] = {
    val (biasGrad, gradient) = delta
    val wStepSize = if (useAdaGrad) stepSize else stepSize / sqrt(iter)
    val l2StepSize = stepSize / sqrt(iter)
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
      val (newW0Grad, newW0Sum, delta) = adaGrad(gradientSum, gradient, 1e-6, 1.0)
      // val (newW0Grad, newW0Sum, delta) = equilibratedGradientDescent(gradientSum, gradient, 1e-8, iter)
      delta.setName(s"delta-$iter").persist(storageLevel).count()

      gradient._2.unpersist(blocking = false)
      val newGradient = delta.mapValues(_._1).filter(_._2 != null).
        setName(s"gradient-$iter").persist(storageLevel)
      newGradient.count()

      if (gradientSum != null) gradientSum._2.unpersist(blocking = false)
      gradientSum = (newW0Sum, delta.mapValues(_._2).setName(s"gradientSum-$iter").persist(storageLevel))
      checkpointGradientSum()
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

  protected def equilibratedGradientDescent(
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

  protected def checkpointGradientSum(): Unit = {
    val sc = gradientSum._2.sparkContext
    if (innerIter % checkpointInterval == 0 && sc.getCheckpointDir.isDefined) {
      gradientSum._2.checkpoint()
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
  val regB: Double,
  val regW: Double,
  val regV: Double,
  val rank: Int,
  val useAdaGrad: Boolean,
  val miniBatchFraction: Double,
  val storageLevel: StorageLevel) extends FM {

  def this(
    input: RDD[(VertexId, LabeledPoint)],
    stepSize: Double = 1e-2,
    regb: Double = 1e-2,
    regw: Double = 1e-2,
    regv: Double = 1e-2,
    rank: Int = 20,
    useAdaGrad: Boolean = true,
    miniBatchFraction: Double = 1.0,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK) {
    this(initializeDataSet(input, rank, storageLevel), stepSize, regb, regw, regv, rank,
      useAdaGrad, miniBatchFraction, storageLevel)
  }

  setDataSet(_dataSet)
  assert(rank > 1, s"rank $rank less than 2")

  override protected def predict(arr: Array[Double]): Double = {
    val result = predictInterval(rank, bias, arr)
    1.0 / (1.0 + math.exp(-result))
  }

  override def saveModel(): FMModel = {
    new FMModel(rank, intercept, true, features)
  }

  override protected def multiplier(q: VertexRDD[VD]): VertexRDD[VD] = {
    dataSet.vertices.leftJoin(q) { (vid, data, deg) =>
      deg match {
        case Some(m) =>
          val y = data.head
          val diff = predict(m) - y
          Array(sumInterval(rank, m), diff)
        case _ => data
      }
    }
  }
}

class FMRegression(
  @transient _dataSet: Graph[VD, ED],
  val stepSize: Double,
  val regB: Double,
  val regW: Double,
  val regV: Double,
  val rank: Int,
  val useAdaGrad: Boolean,
  val miniBatchFraction: Double,
  val storageLevel: StorageLevel) extends FM {

  def this(
    input: RDD[(VertexId, LabeledPoint)],
    stepSize: Double = 1e-2,
    regb: Double = 1e-2,
    regw: Double = 1e-2,
    regv: Double = 1e-2,
    rank: Int = 20,
    useAdaGrad: Boolean = true,
    miniBatchFraction: Double = 1.0,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK) {
    this(initializeDataSet(input, rank, storageLevel), stepSize, regb, regw, regv, rank,
      useAdaGrad, miniBatchFraction, storageLevel)
  }

  setDataSet(_dataSet)
  assert(rank > 1, s"rank $rank less than 2")

  // val max = samples.map(_._2.head).max
  // val min = samples.map(_._2.head).min

  override protected def predict(arr: Array[Double]): Double = {
    var result = predictInterval(rank, bias, arr)
    // result = Math.max(result, min)
    // result = Math.min(result, max)
    result
  }

  override protected def multiplier(q: VertexRDD[VD]): VertexRDD[VD] = {
    dataSet.vertices.leftJoin(q) { (vid, data, deg) =>
      deg match {
        case Some(m) =>
          val y = data.head
          // + Utils.random.nextGaussian()
          val diff = predict(m) - y
          Array(sumInterval(rank, m), diff * 2.0)
        case _ => data
      }
    }
  }
}

object FM {
  private[ml] type ED = Double
  private[ml] type VD = Array[Double]

  /**
   * FM 分类
   * @param input 训练数据
   * @param numIterations 迭代次数
   * @param stepSize  学习步长推荐 1e-2- 1e-1
   * @param regb   L2范数作用于公式中的 w_{0} 部分
   * @param regw   L2范数作用于公式中的 \sum_{j=1}^{n}w_{j}x_{j} 部分
   * @param regv   L2范数作用于公式中的 \sum_{i=1}^{n}\sum_{j=i+1}^{n}<v_{i},v_{j}> x_{i}x_{j} 部分
   * @param rank   特征分解向量的维度推荐 10-20
   * @param useAdaGrad 使用 AdaGrad训练
   * @param miniBatchFraction  每次迭代采样比例
   * @param storageLevel   缓存级别
   * @return
   */
  def trainClassification(
    input: RDD[(Long, LabeledPoint)],
    numIterations: Int,
    stepSize: Double,
    regb: Double,
    regw: Double,
    regv: Double,
    rank: Int,
    useAdaGrad: Boolean = true,
    miniBatchFraction: Double = 1.0,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK): FMModel = {
    val data = input.map { case (id, LabeledPoint(label, features)) =>
      assert(id >= 0.0, s"sampleId $id less than 0")
      val newLabel = if (label > 0.0) 1.0 else 0.0
      (id, LabeledPoint(newLabel, features))
    }
    val lfm = new FMClassification(data, stepSize, regb, regw, regv, rank,
      useAdaGrad, miniBatchFraction, storageLevel)
    lfm.run(numIterations)
    val model = lfm.saveModel()
    model
  }

  /**
   * FM 回归
   * @param input 训练数据
   * @param numIterations 迭代次数
   * @param stepSize  学习步长推荐 1e-2- 1e-1
   * @param regb   L2范数作用于公式中的 w_{0} 部分
   * @param regw   L2范数作用于公式中的 \sum_{j=1}^{n}w_{j}x_{j} 部分
   * @param regv   L2范数作用于公式中的 \sum_{i=1}^{n}\sum_{j=i+1}^{n}<v_{i},v_{j}> x_{i}x_{j} 部分
   * @param rank   特征分解向量的维度推荐 10-20
   * @param useAdaGrad 使用 AdaGrad训练
   * @param miniBatchFraction  每次迭代采样比例
   * @param storageLevel   缓存级别
   * @return
   */

  def trainRegression(
    input: RDD[(Long, LabeledPoint)],
    numIterations: Int,
    stepSize: Double,
    regb: Double,
    regw: Double,
    regv: Double,
    rank: Int,
    useAdaGrad: Boolean = true,
    miniBatchFraction: Double = 1.0,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK): FMModel = {
    val data = input.map { case (id, labeledPoint) =>
      assert(id >= 0.0, s"sampleId $id less than 0")
      (id, labeledPoint)
    }
    val lfm = new FMRegression(data, stepSize, regb, regw, regv, rank,
      useAdaGrad, miniBatchFraction, storageLevel)
    lfm.run(numIterations)
    val model = lfm.saveModel()
    model
  }

  private[ml] def initializeDataSet(
    input: RDD[(VertexId, LabeledPoint)],
    rank: Int,
    storageLevel: StorageLevel): Graph[VD, ED] = {
    val edges = input.flatMap { case (sampleId, labelPoint) =>
      val newId = newSampleId(sampleId) // sample id
      labelPoint.features.activeIterator.filter(_._2 != 0.0).map { case (index, value) =>
        Edge(index, newId, value)
      }
    }
    val vertices = input.map { case (sampleId, labelPoint) =>
      val newId = newSampleId(sampleId)
      (newId, labelPoint.label)
    }
    val dataSet = Graph.fromEdges(edges, null.asInstanceOf[VD], storageLevel, storageLevel)
    dataSet.outerJoinVertices(vertices) { (vid, data, deg) =>
      deg match {
        case Some(i) => {
          Array.fill(1)(i) // label point
        }
        case None => {
          val a = Array.fill(rank + 1)(Utils.random.nextGaussian() * 1e-2) // parameter point
          a
        }
      }
    }
  }

  private[ml] def newSampleId(id: Long): VertexId = {
    -(id + 1L)
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

  private[ml] def forwardReduceInterval(a: VD, b: VD): VD = {
    var i = 0
    while (i < a.length) {
      a(i) += b(i)
      i += 1
    }
    a
  }

  /**
   * arr[0] = w_{i}x{i}
   * arr[f] = v_{i,j}x_{i} f属于 [1,rank]
   * arr[k] = v_{i,j}^{2}x_{i}^{2} k属于 (rank,rank * 2 + 1]
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

