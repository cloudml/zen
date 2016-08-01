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

import com.fasterxml.jackson.core.JsonParser.Feature
import com.github.cloudml.zen.ml.util.Utils
import org.apache.spark.{Logging, SparkContext}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vector => SV}
import breeze.linalg.{SparseVector => BSV}
import org.apache.spark.storage.StorageLevel

import com.github.cloudml.zen.ml.util.SparkUtils._
import com.github.cloudml.zen.ml.recommendation.MVM._
import scala.math._

/**
  * Multi-view Machines :
  * \hat{y}(x) :=\sum_{i_1 =1}^{I_i +1} ...\sum_{i_m =1}^{I_m +1}
  * (\prod_{v=1}^{m} z_{i_v}^{(v)})(\sum_{f=1}^{k}\prod_{v=1}^{m}a_{i_{v,j}}^{(v)})
  * :=  \sum_{f}^{k}(\sum_{i_1 =1}^{I_1+1}z_{i_1}^{(1)}a_{i_1,j}^{(1)}) ..
  * (\sum_{i_m =1}^{I_m+1}z_{i_m}^{(m)}a_{i_m,j}^{(m)})
  *
  * derivative of the model :
  * \frac{\partial \hat{y}(x|\Theta )}{\partial\theta} :=z_{i_{v}}^{(v)}
  * (\sum_{i_1 =1}^{I_1+1}z_{i_1}^{(1)}a_{i_1,j}^{(1)}) ...
  * (\sum_{i_{v-1} =1}^{I_{v-1}+1}z_{i_{v-1}}^{({v-1})}a_{i_{v-1},j}^{({v-1})})
  * (\sum_{i_{v+1} =1}^{I_{v+1}+1}z_{i_{v+1}}^{({v+1})}a_{i_{v+1},j}^{({v+1})}) ...
  * (\sum_{i_m =1}^{I_m+1}z_{i_m}^{(m)}a_{i_m,j}^{(m)})
  */
private[ml] abstract class VecMVM extends Serializable with Logging {

  def rank: Int

  def storageLevel: StorageLevel

  def stepSize: Double

  def views: Array[Long]

  def dataSet: RDD[(Long, LabeledPoint)]

  def miniBatchFraction: Double

  def featureSize: Int

  def factors: Array[Array[Double]]

  def elasticNetParam: Double

  def regParam: Double

  protected[ml] def mask: Int = {
    max(1 / miniBatchFraction, 1).toInt
  }

  def extDataSet: RDD[(Long, LabeledPoint)] = dataSet.map(x => {
    val start = views.last.toInt
    val sv = BSV.zeros[Double](views.length + views.last.toInt)
    for(i <- views.indices){
      sv(start + i) = 1.0
    }
    x._2.features.activeIterator.foreach(y => sv(y._1) = y._2)
    (x._1,new LabeledPoint(x._2.label, sv))
  }).persist(storageLevel)
  extDataSet.count()

  def run(iterations: Int): Unit = {
    var his = Array.fill(featureSize + views.length, rank)(0.0)
    for (iter <- 1 to iterations) {
      logInfo(s"Start train (Iteration $iter/$iterations)")
      val mod = mask
      val random = genRandom(mod, iter)
      val seed = random.nextLong()
      val broadcastData = extDataSet.context.broadcast(factors)
      val startedAt = System.nanoTime()
      val out = extDataSet.mapPartitions(it => {
        val facs = broadcastData.value
        var grad = Array.fill(featureSize + views.length, rank)(0.0)
        var trainingLoss = 0.0
        var numSamples = 0
        while(it.hasNext){
          val x = it.next()
          if(mod == 1 || isSampled(random, seed, x._1, iter, mod)) {
            val out = getGradient(rank, x._2, facs, views, grad)
            grad = out._1
            trainingLoss += out._2
            numSamples += 1
            }
        }
        Array((numSamples, grad, trainingLoss)).toIterator
      }).reduce(VecMVM.reduceGrad)
      val gradients = out._2.map(x => x.map(_ / out._1))
      val elapsedSeconds = (System.nanoTime() - startedAt) / 1e9
      logInfo(s"Training loss : ${getLoss(out)}")
      logInfo(s"End  train (Iteration $iter/$iterations) takes:         $elapsedSeconds")

      his = VecMVM.addGrad(his, gradients)

      updateWeight(his, gradients, factors)
    }
  }

  def getLoss(out: (Int, Array[Array[Double]], Double)) : Double

  def getGradient(rank: Int, vec: LabeledPoint, factors: Array[Array[Double]],
                  views: Array[Long], grads: Array[Array[Double]]): (Array[Array[Double]], Double)

  def updateWeight(his: Array[Array[Double]], grad: Array[Array[Double]],
                     factors: Array[Array[Double]]): Array[Array[Double]] ={
    val eps = 1e-6
    val regParamL2 = (1.0 - elasticNetParam) * regParam
    val shrinkageVal = elasticNetParam * regParam * stepSize
    for(i <- his.indices) {
      for(j <- his(0).indices) {
        val newGrad = grad(i)(j) / (math.sqrt(his(i)(j)) + eps)
        if(newGrad != 0.0) {
          factors(i)(j) -= stepSize * (newGrad + factors(i)(j) * regParamL2)
          if(shrinkageVal > 0) {
            factors(i)(j) = math.signum(factors(i)(j)) * math.max(0.0, abs(factors(i)(j) - shrinkageVal))
          }
        }
      }
    }
    factors
  }

  def init(rank: Int, views: Array[Long]): Array[Array[Double]] = {
    val arr = 0.toLong until  (views.last + views.length)
    arr.map(x => {
        Array.fill(rank) {
          Utils.random.nextGaussian() * 1e-2
      }
    }).toArray
  }

  def saveModel(): MVMModel = {
    new MVMModel(rank, views, false,
      dataSet.context.parallelize(factors.zipWithIndex.map(x => (x._2.toLong, x._1))))
  }

}

class VecMVMRegression(val rank: Int,
                       val stepSize: Double,
                       val regParam: Double,
                       val views: Array[Long],
                       val dataSet: RDD[(Long, LabeledPoint)],
                       val miniBatchFraction: Double,
                       val featureSize: Int,
                       val elasticNetParam: Double,
                       val storageLevel: StorageLevel) extends VecMVM {
  val factors: Array[Array[Double]] = init(rank, views)

  def getGradient(rank: Int, vec: LabeledPoint, factors: Array[Array[Double]],
                  views: Array[Long], grads: Array[Array[Double]]): (Array[Array[Double]], Double) = {
    val vSize = views.length
    val ms = VecMVM.sumInterval(rank, vec.features, factors, views)
    val multi= VecMVM.multiplier(rank, views.length, ms)
    val ybar = multi.sum
    val diff = math.pow(ybar - vec.label, 2)
    vec.features.activeIterator.foreach(x => {
      val index = x._1
      val value = x._2
      val vid = featureId2viewId(index, views)
      for(i <- 0 until rank) {
        val delta = if(ms(i*vSize + vid) == 0.0) 0.0 else multi(i) / ms(i*vSize + vid)
        grads(index)(i) += (2.0 * (ybar - vec.label)* value * delta)
      }
    })
    (grads, diff)
  }

  def getLoss(out: (Int, Array[Array[Double]], Double)) : Double = {
    math.sqrt(out._3 / out._1)
  }
}

class VecMVMClassification(val rank: Int,
                       val stepSize: Double,
                       val regParam: Double,
                       val views: Array[Long],
                       val dataSet: RDD[(Long, LabeledPoint)],
                       val miniBatchFraction: Double,
                       val featureSize: Int,
                       val elasticNetParam: Double,
                       val storageLevel: StorageLevel) extends VecMVM {
  val factors: Array[Array[Double]] = init(rank, views)

  @inline private def sigmoid(x: Double): Double = {
    1d / (1d + math.exp(-x))
  }

  def getGradient(rank: Int, vec: LabeledPoint, factors: Array[Array[Double]],
                  views: Array[Long], grads: Array[Array[Double]]): (Array[Array[Double]], Double) = {
    val vSize = views.length
    val ms = VecMVM.sumInterval(rank, vec.features, factors, views)
    val multi= VecMVM.multiplier(rank, views.length, ms)
    val ybar = multi.sum
    val diff = Utils.log1pExp(if (vec.label > 0.0) -ybar else ybar)
    vec.features.activeIterator.foreach(x => {
      val index = x._1
      val value = x._2
      val vid = featureId2viewId(index, views)
      for(i <- 0 until rank) {
        val delta = if(ms(i*vSize + vid) == 0.0) 0.0 else multi(i) / ms(i*vSize + vid)
        grads(index)(i) += (-vec.label * sigmoid(-vec.label * ybar))* value * delta
      }
    })
    (grads, diff)
  }

  def getLoss(out: (Int, Array[Array[Double]], Double)) : Double = {
    out._3 / out._1
  }
  override def saveModel(): MVMModel = {
    new MVMModel(rank, views, true,
      dataSet.context.parallelize(factors.zipWithIndex.map(x => (x._2.toLong, x._1))))
  }
}

object VecMVM {
  def sumInterval(rank: Int, vec: SV, factors: Array[Array[Double]],
                  views: Array[Long]): Array[Double] = {
    val vSize = views.length
    val out = Array.fill(rank*vSize)(0.0)
    vec.activeIterator.foreach(x => {
      val index = x._1
      val z = x._2
      val vid = featureId2viewId(index, views)
      for(i <- 0 until rank) {
        out(i*vSize + vid) += z*factors(index)(i)
      }
    })
    out
  }

  def multiplier(rank: Int, vSize: Int, arr: Array[Double]): Array[Double] = {
    val out = Array.fill(rank)(1.0)
    for(i <- arr.indices) {
      out(i/vSize) *= arr(i)
    }
    out
  }

  def reduceGrad(a: (Int, Array[Array[Double]], Double), b: (Int, Array[Array[Double]], Double)):
  (Int, Array[Array[Double]], Double) = {
    val x = a._2.length
    val y = a._2(0).length
    val out = Array.fill(x, y)(0.0)
    for(i <- 0 until x) {
      for(j <- 0 until y) {
        out(i)(j) = a._2(i)(j) + b._2(i)(j)
      }
    }
    (a._1 + b._1, out, a._3 + b._3)
  }

  def addGrad(a: Array[Array[Double]], b: Array[Array[Double]]): Array[Array[Double]] = {
    val x = a.length
    val y = a(0).length
    for(i <- 0 until x) {
      for(j <- 0 until y) {
        a(i)(j) += math.pow(b(i)(j), 2)
      }
    }
    a
  }
}
