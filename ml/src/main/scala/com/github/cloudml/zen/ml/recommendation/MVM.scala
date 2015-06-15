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
import com.github.cloudml.zen.ml.recommendation.MVM._
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
 * Multi-view Machines 公式定义:
 * \hat{y}(x) :=\sum_{i_1 =1}^{I_i +1} ...\sum_{i_m =1}^{I_m +1}
 * (\prod_{v=1}^{m} z_{i_v}^{(v)})(\sum_{f=1}^{k}\prod_{v=1}^{m}a_{i_{v,j}}^{(v)})
 * :=  \sum_{f}^{k}(\sum_{i_1 =1}^{I_1+1}z_{i_1}^{(1)}a_{i_1,j}^{(1)}) ..
 * (\sum_{i_m =1}^{I_m+1}z_{i_m}^{(m)}a_{i_m,j}^{(m)})
 *
 * 其导数是:
 * \frac{\partial \hat{y}(x|\Theta )}{\partial\theta} :=z_{i_{v}}^{(v)}
 * (\sum_{i_1 =1}^{I_1+1}z_{i_1}^{(1)}a_{i_1,j}^{(1)}) ...
 * (\sum_{i_{v-1} =1}^{I_{v-1}+1}z_{i_{v-1}}^{({v-1})}a_{i_{v-1},j}^{({v-1})})
 * (\sum_{i_{v+1} =1}^{I_{v+1}+1}z_{i_{v+1}}^{({v+1})}a_{i_{v+1},j}^{({v+1})}) ...
 * (\sum_{i_m =1}^{I_m+1}z_{i_m}^{(m)}a_{i_m,j}^{(m)})
 */
private[ml] abstract class MVM extends Serializable with Logging {

  protected val checkpointInterval = 10
  protected var numFeatures: Long = 0
  protected var numSamples: Long = 0

  // SGD
  @transient protected var dataSet: Graph[VD, ED] = null
  @transient protected var multi: VertexRDD[Array[Double]] = null
  @transient protected var gradientSum: VertexRDD[Array[Double]] = null
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

  def views: Array[Long]

  def regL2: Double

  def rank: Int

  def useAdaGrad: Boolean

  def storageLevel: StorageLevel

  def miniBatchFraction: Double

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
      logInfo(s"Train (Iteration $iter/$iterations) cost:               ${loss(margin)}")
      logInfo(s"End  train (Iteration $iter/$iterations) takes:         $elapsedSeconds")

      previousVertices.unpersist(blocking = false)
      margin.unpersist(blocking = false)
      multi.unpersist(blocking = false)
      gradient.unpersist(blocking = false)
      innerIter += 1
    }
  }

  def saveModel(): MVMModel = {
    new MVMModel(rank, views, false, features)
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

  protected def backward(q: VertexRDD[VD], iter: Int): VertexRDD[Array[Double]] = {
    val thisNumSamples = (1.0 / mask) * numSamples
    multi = multiplier(q).setName(s"multiplier-$iter").persist(storageLevel)
    val sampledArrayLen = rank * views.length + 1
    val gradient = GraphImpl.fromExistingRDDs(multi, edges).aggregateMessages[Array[Double]](ctx => {
      // val sampleId = ctx.dstId
      val featureId = ctx.srcId
      if (ctx.dstAttr.length == sampledArrayLen) {
        val x = ctx.attr
        val arr = ctx.dstAttr
        val viewId = featureId2viewId(featureId, views)
        val m = backwardInterval(rank, viewId, x, arr, arr.last)
        // send the multi directly
        ctx.sendToSrc(m)
      }
    }, forwardReduceInterval, TripletFields.Dst).mapValues { gradients =>
      gradients.map(_ / thisNumSamples)
    }
    gradient.setName(s"gradient-$iter").persist(storageLevel)

  }

  // Updater for L2 regularized problems
  protected def updateWeight(delta: VertexRDD[Array[Double]], iter: Int): VertexRDD[VD] = {
    val gradient = delta
    val wStepSize = if (useAdaGrad) stepSize else stepSize / sqrt(iter)
    val l2StepSize = stepSize / sqrt(iter)

    dataSet.vertices.leftJoin(gradient) { (_, attr, gradient) =>
      gradient match {
        case Some(grad) =>
          val weight = attr
          var i = 0
          while (i < rank) {
            weight(i) -= wStepSize * grad(i) + l2StepSize * regL2 * weight(i)
            i += 1
          }
          weight
        case None => attr
      }
    }.setName(s"vertices-$iter").persist(storageLevel)
  }

  protected def updateGradientSum(
    gradient: VertexRDD[Array[Double]],
    iter: Int): VertexRDD[Array[Double]] = {
    if (useAdaGrad) {
      val delta = adaGrad(gradientSum, gradient, 1e-6, 1.0)
      // val delta = equilibratedGradientDescent(gradientSum, gradient, 1e-4, iter)
      delta.setName(s"delta-$iter").persist(storageLevel).count()

      gradient.unpersist(blocking = false)
      val newGradient = delta.mapValues(_._1).filter(_._2 != null).
        setName(s"gradient-$iter").persist(storageLevel)
      newGradient.count()

      if (gradientSum != null) gradientSum.unpersist(blocking = false)
      gradientSum = delta.mapValues(_._2).setName(s"gradientSum-$iter").persist(storageLevel)
      checkpointGradientSum()
      gradientSum.count()
      delta.unpersist(blocking = false)
      newGradient
    } else {
      gradient
    }
  }

  protected def adaGrad(
    gradientSum: VertexRDD[Array[Double]],
    gradient: VertexRDD[Array[Double]],
    epsilon: Double,
    rho: Double): VertexRDD[(Array[Double], Array[Double])] = {
    val delta = if (gradientSum == null) {
      features.mapValues(t => t.map(x => 0.0))
    }
    else {
      gradientSum
    }
    val newGradSum = delta.leftJoin(gradient) { (_, gradSum, g) =>
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
    newGradSum
  }

  protected def equilibratedGradientDescent(
    gradientSum: VertexRDD[Array[Double]],
    gradient: VertexRDD[Array[Double]],
    epsilon: Double,
    iter: Int): VertexRDD[(Array[Double], Array[Double])] = {
    val delta = if (gradientSum == null) {
      features.mapValues(t => t.map(x => 0.0))
    }
    else {
      gradientSum
    }
    val newGradSum = delta.leftJoin(gradient) { (_, gradSum, g) =>
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
    newGradSum
  }

  protected def checkpointGradientSum(): Unit = {
    val sc = gradientSum.sparkContext
    if (innerIter % checkpointInterval == 0 && sc.getCheckpointDir.isDefined) {
      gradientSum.checkpoint()
    }
  }

  protected def checkpointVertices(): Unit = {
    val sc = vertices.sparkContext
    if (innerIter % checkpointInterval == 0 && sc.getCheckpointDir.isDefined) {
      vertices.checkpoint()
    }
  }
}

class MVMClassification(
  @transient _dataSet: Graph[VD, ED],
  val stepSize: Double,
  val views: Array[Long],
  val regL2: Double,
  val rank: Int,
  val useAdaGrad: Boolean,
  val miniBatchFraction: Double,
  val storageLevel: StorageLevel) extends MVM {

  def this(
    input: RDD[(VertexId, LabeledPoint)],
    stepSize: Double = 1e-2,
    views: Array[Long],
    regL2: Double = 1e-3,
    rank: Int = 20,
    useAdaGrad: Boolean = true,
    miniBatchFraction: Double = 1.0,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK) {
    this(initializeDataSet(input, views, rank, storageLevel), stepSize, views, regL2, rank,
      useAdaGrad, miniBatchFraction, storageLevel)
  }

  setDataSet(_dataSet)
  assert(rank > 1, s"rank $rank less than 2")

  override protected def predict(arr: Array[Double]): Double = {
    val result = predictInterval(rank, arr)
    1.0 / (1.0 + math.exp(-result))
  }

  override def saveModel(): MVMModel = {
    new MVMModel(rank, views, true, features)
  }

  override protected def multiplier(q: VertexRDD[VD]): VertexRDD[VD] = {
    dataSet.vertices.leftJoin(q) { (vid, data, deg) =>
      deg match {
        case Some(m) =>
          val y = data.head
          val diff = predict(m) - y
          val ret = sumInterval(rank, m)
          ret(ret.length - 1) = diff
          ret
        case _ => data
      }
    }
  }
}

class MVMRegression(
  @transient _dataSet: Graph[VD, ED],
  val stepSize: Double,
  val views: Array[Long],
  val regL2: Double,
  val rank: Int,
  val useAdaGrad: Boolean,
  val miniBatchFraction: Double,
  val storageLevel: StorageLevel) extends MVM {

  def this(
    input: RDD[(VertexId, LabeledPoint)],
    stepSize: Double = 1e-2,
    views: Array[Long],
    regL2: Double,
    rank: Int = 20,
    useAdaGrad: Boolean = true,
    miniBatchFraction: Double = 1.0,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK) {
    this(initializeDataSet(input, views, rank, storageLevel), stepSize, views, regL2, rank,
      useAdaGrad, miniBatchFraction, storageLevel)
  }

  setDataSet(_dataSet)
  assert(rank > 1, s"rank $rank less than 2")

  // val max = samples.map(_._2.head).max
  // val min = samples.map(_._2.head).min

  override protected def predict(arr: Array[Double]): Double = {
    var result = predictInterval(rank, arr)
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
          val ret = sumInterval(rank, m)
          ret(ret.length - 1) = diff * 2.0
          ret
        case _ => data
      }
    }
  }
}

object MVM {
  private[ml] type ED = Double
  private[ml] type VD = Array[Double]

  /**
   * MVM 分类
   * @param input 训练数据
   * @param numIterations 迭代次数
   * @param stepSize  学习步长推荐 1e-2- 1e-1
   * @param regL2   L2范数
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
    views: Array[Long],
    regL2: Double,
    rank: Int,
    useAdaGrad: Boolean = true,
    miniBatchFraction: Double = 1.0,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK): MVMModel = {
    val data = input.map { case (id, LabeledPoint(label, features)) =>
      assert(id >= 0.0, s"sampleId $id less than 0")
      val newLabel = if (label > 0.0) 1.0 else 0.0
      (id, LabeledPoint(newLabel, features))
    }
    val lfm = new MVMClassification(data, stepSize, views, regL2, rank,
      useAdaGrad, miniBatchFraction, storageLevel)
    lfm.run(numIterations)
    val model = lfm.saveModel()
    model
  }

  /**
   * MVM 回归
   * @param input 训练数据
   * @param numIterations 迭代次数
   * @param stepSize  学习步长推荐 1e-2- 1e-1
   * @param regL2   L2范数
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
    views: Array[Long],
    regL2: Double,
    rank: Int,
    useAdaGrad: Boolean = true,
    miniBatchFraction: Double = 1.0,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK): MVMModel = {
    val data = input.map { case (id, labeledPoint) =>
      assert(id >= 0.0, s"sampleId $id less than 0")
      (id, labeledPoint)
    }
    val lfm = new MVMRegression(data, stepSize, views, regL2, rank,
      useAdaGrad, miniBatchFraction, storageLevel)
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
      labelPoint.features.activeIterator.filter(_._2 != 0.0).map { case (featureId, value) =>
        assert(featureId < numFeatures)
        Edge(featureId, newId, value)
      } ++ views.indices.map { i => Edge(numFeatures + i, newId, 1D) }
    }.persist(storageLevel)
    edges.count()

    val vertices = (input.map { case (sampleId, labelPoint) =>
      val newId = newSampleId(sampleId)
      val label = Array(labelPoint.label)
      // label point
      (newId, label)
    } ++ edges.map(_.srcId).distinct().map { featureId =>
      // parameter point
      val parms = Array.fill(rank) {
        Utils.random.nextGaussian() * 0.1
      }
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
   * arr(k + v * rank)= (\sum_{i_1 =1}^{I_1+1}z_{i_1}^{(1)}a_{i_1,j}^{(1)}) ..
   * (\sum_{i_m =1}^{I_m+1}z_{i_m}^{(m)}a_{i_m,j}^{(m)})
   */
  private[ml] def predictInterval(rank: Int, arr: VD): ED = {
    val viewSize = arr.length / rank
    var sum = 0.0
    var i = 0
    while (i < rank) {
      var multi = 1.0
      var viewId = 0
      while (viewId < viewSize) {
        multi *= arr(i + viewId * rank)
        viewId += 1
      }
      sum += multi
      i += 1
    }
    sum
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
    val arr = new Array[Double](rank * viewSize)
    forwardInterval(rank, viewId, arr, x, w)
  }

  /**
   * arr的长度是rank * viewSize
   * f属于 [viewId * rank ,(viewId +1) * rank)时
   * arr[f] = z_{i_v}^{(1)}a_{i_{i},j}^{(1)}
   */
  private[ml] def forwardInterval(rank: Int, viewId: Int, arr: Array[Double], z: ED, w: VD): VD = {
    var i = 0
    while (i < rank) {
      arr(i + viewId * rank) += z * w(i)
      i += 1
    }
    arr
  }

  /**
   * arr的长度是rank * viewSize
   * 当 v=viewId , k属于[0,rank] 时
   * arr(k + v * rank) = \frac{\partial \hat{y}(x|\Theta )}{\partial\theta }
   * 返回 multi * \frac{\partial \hat{y}(x|\Theta )}{\partial\theta }
   * 分类: multi = 1/(1+ \exp(-\hat{y}(x|\Theta)) ) - y
   * 回归: multi = 2(\hat{y}(x|\Theta) -y)
   */
  private[ml] def backwardInterval(
    rank: Int,
    viewId: Int,
    x: ED,
    arr: VD,
    multi: ED): VD = {
    val m = new Array[Double](rank)
    var i = 0
    while (i < rank) {
      m(i) = multi * x * arr(i + viewId * rank)
      i += 1
    }
    m
  }

  /**
   * arr(k + v * rank)= (\sum_{i_1 =1}^{I_1+1}z_{i_1}^{(1)}a_{i_1,j}^{(1)}) ..
   * (\sum_{i_m =1}^{I_m+1}z_{i_m}^{(m)}a_{i_m,j}^{(m)})
   */
  private[ml] def sumInterval(rank: Int, arr: Array[Double]): VD = {
    val viewSize = arr.length / rank
    val m = new Array[Double](rank)
    var i = 0
    while (i < rank) {
      var multi = 1.0
      var viewId = 0
      while (viewId < viewSize) {
        multi *= arr(i + viewId * rank)
        viewId += 1
      }
      m(i) += multi
      i += 1
    }

    val ret = new Array[Double](rank * viewSize + 1)
    i = 0
    while (i < rank) {
      var viewId = 0
      while (viewId < viewSize) {
        ret(i + viewId * rank) = m(i) / arr(i + viewId * rank)
        viewId += 1
      }

      i += 1
    }
    ret
  }
}

