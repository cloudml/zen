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

package com.github.cloudml.zen.examples.ml

import breeze.linalg.{SparseVector => BSV}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{SparseVector => SSV}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

object AdsUtils {
  def genSamplesWithTime(
    sc: SparkContext,
    input: String,
    numPartitions: Int = -1,
    sampleFraction: Double = 1.0,
    newLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK):
  (RDD[(Long, LabeledPoint)], RDD[(Long, LabeledPoint)], Array[Long]) = {
    var adaDataSet = sc.textFile(input, sc.defaultParallelism).map { line =>
      val arr = line.split(" ")
      val Array(label, importance) = arr.head.split(":")
      val features = arr.tail.map { sub =>
        val Array(featureId, featureValue, viewId) = sub.split(":")
        (featureId.toInt, featureValue.toDouble, viewId.toInt)
      }
      (importance.toInt, label.toDouble, features)
    }
    if (sampleFraction < 1) adaDataSet = adaDataSet.sample(false, sampleFraction)
    adaDataSet = if (numPartitions > 0) {
      adaDataSet.repartition(numPartitions)
    } else {
      adaDataSet.repartition(sc.defaultParallelism)
    }
    adaDataSet.persist(newLevel).count()

    val f2v = adaDataSet.flatMap { case (_, _, features) =>
      features.map { case (featureId, featureValue, viewId) =>
        (featureId, viewId)
      }
    }.distinct().collect()

    val viewIds = f2v.map(_._2).distinct
    val v2f2i = viewIds.map { viewId =>
      val f2i = f2v.filter(_._2 == viewId).map(_._1).zipWithIndex.toMap
      (viewId, f2i)
    }.toMap

    val maxUnigramsId = v2f2i(1).size + 1
    val maxDisplayUrlId = v2f2i(2).size + 1
    val maxPositionId = v2f2i(3).size + 1
    val maxMatchTypeId = v2f2i(4).size + 1
    val numFeatures = maxUnigramsId + maxDisplayUrlId + maxPositionId + maxMatchTypeId

    val dataSet = adaDataSet.map { case (_, label, features) =>
      val sv = BSV.zeros[Double](numFeatures)
      features.foreach { case (featureId, featureValue, viewId) =>
        val offset = if (viewId == 1) {
          v2f2i(1)(featureId)
        } else if (viewId == 2) {
          maxUnigramsId + v2f2i(2)(featureId)
        } else if (viewId == 3) {
          maxUnigramsId + maxDisplayUrlId + v2f2i(3)(featureId)
        } else if (viewId == 4) {
          maxUnigramsId + maxDisplayUrlId + maxPositionId + v2f2i(4)(featureId)
        } else {
          throw new IndexOutOfBoundsException("viewID must be less than 5")
        }
        sv(offset) = featureValue
      }
      new LabeledPoint(label, new SSV(sv.length, sv.index.slice(0, sv.used), sv.data.slice(0, sv.used)))

    }.zipWithIndex().map(_.swap).persist(StorageLevel.MEMORY_AND_DISK)

    val testSet = dataSet.filter(_._2.features.hashCode() % 5 == 0)
    val trainSet = dataSet.filter(_._2.features.hashCode() % 5 != 0)
    trainSet.count()
    testSet.count()
    adaDataSet.unpersist()
    /**
     * [0,maxUnigramsId), [maxUnigramsId + 1,maxDisplayUrlId), [maxDisplayUrlId + 1,maxPositionId)
     * [maxPositionId + 1, numFeatures)
     */
    val views = Array(maxUnigramsId + 1, maxDisplayUrlId + 1, maxPositionId + 1, numFeatures).map(_.toLong)
    (trainSet, testSet, views)
  }

  def genSamplesWithTimeAnd3Views(
    sc: SparkContext,
    input: String,
    numPartitions: Int = -1,
    sampleFraction: Double = 1.0,
    newLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK):
  (RDD[(Long, LabeledPoint)], RDD[(Long, LabeledPoint)], Array[Long]) = {
    var adaDataSet = sc.textFile(input, sc.defaultParallelism).map { line =>
      val arr = line.split(" ")
      val Array(label, importance) = arr.head.split(":")
      val features = arr.tail.map { sub =>
        val Array(featureId, featureValue, viewId) = sub.split(":")
        (featureId.toInt, featureValue.toDouble, math.min(viewId.toInt, 3))
      }
      (importance.toInt, label.toDouble, features)
    }
    if (sampleFraction < 1) adaDataSet = adaDataSet.sample(false, sampleFraction)
    adaDataSet = if (numPartitions > 0) {
      adaDataSet.repartition(numPartitions)
    } else {
      adaDataSet.repartition(sc.defaultParallelism)
    }
    adaDataSet.persist(newLevel).count()

    val f2v = adaDataSet.flatMap { case (_, _, features) =>
      features.map { case (featureId, featureValue, viewId) =>
        (featureId, viewId)
      }
    }.distinct().collect()

    val viewIds = f2v.map(_._2).distinct
    val v2f2i = viewIds.map { viewId =>
      val f2i = f2v.filter(_._2 == viewId).map(_._1).zipWithIndex.toMap
      (viewId, f2i)
    }.toMap

    val maxUnigramsId = v2f2i(1).size + 1
    val maxDisplayUrlId = v2f2i(2).size + 1
    val maxPositionId = v2f2i(3).size + 1
    val numFeatures = maxUnigramsId + maxDisplayUrlId + maxPositionId
    val dataSet = adaDataSet.map { case (_, label, features) =>
      val sv = BSV.zeros[Double](numFeatures)
      features.foreach { case (featureId, featureValue, viewId) =>
        val offset = if (viewId == 1) {
          v2f2i(1)(featureId)
        } else if (viewId == 2) {
          maxUnigramsId + v2f2i(2)(featureId)
        } else if (viewId == 3) {
          maxUnigramsId + maxDisplayUrlId + v2f2i(3)(featureId)
        } else {
          throw new IndexOutOfBoundsException("viewID must be less than 4")
        }
        sv(offset) = featureValue
      }
      new LabeledPoint(label, new SSV(sv.length, sv.index.slice(0, sv.used), sv.data.slice(0, sv.used)))

    }.zipWithIndex().map(_.swap).persist(StorageLevel.MEMORY_AND_DISK)

    val testSet = dataSet.filter(_._2.features.hashCode() % 5 == 0)
    val trainSet = dataSet.filter(_._2.features.hashCode() % 5 != 0)
    trainSet.count()
    testSet.count()
    adaDataSet.unpersist()
    /**
     * [0,maxUnigramsId), [maxUnigramsId + 1,maxDisplayUrlId), [maxDisplayUrlId + 1,maxPositionId)
     * [maxPositionId + 1, numFeatures)
     */
    val views = Array(maxUnigramsId + 1, maxDisplayUrlId + 1, maxPositionId + 1, numFeatures).map(_.toLong)
    (trainSet, testSet, views)
  }

}
