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
import com.github.cloudml.zen.ml.util.Logging
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{SparseVector => SSV}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

private[zen] object MovieLensUtils extends Logging {

  def genSamplesWithTime(
    sc: SparkContext,
    dataFile: String,
    numPartitions: Int = -1,
    newLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK):
  (RDD[(Long, LabeledPoint)], RDD[(Long, LabeledPoint)], Array[Long]) = {
    val line = sc.textFile(dataFile).first()
    val splitString = if (line.contains(",")) "," else "::"
    var movieLens = sc.textFile(dataFile, sc.defaultParallelism).mapPartitions { iter =>
      iter.filter(t => !t.startsWith("userId") && !t.isEmpty).map { line =>
        val Array(userId, movieId, rating, timestamp) = line.split(splitString)
        (userId.toInt, movieId.toInt, rating.toDouble, timestamp.toInt)
      }
    }
    movieLens = movieLens.repartition(if (numPartitions > 0) numPartitions else sc.defaultParallelism)
    movieLens.persist(newLevel).count()

    val daySeconds = 60 * 60 * 24
    val maxUserId = movieLens.map(_._1).max + 1
    val maxMovieId = movieLens.map(_._2).max + 1
    val maxTime = movieLens.map(_._4 / daySeconds).max()
    val minTime = movieLens.map(_._4 / daySeconds).min()
    val maxDay = maxTime - minTime + 1
    val numFeatures = maxUserId + maxMovieId + maxDay

    val dataSet = movieLens.map { case (userId, movieId, rating, timestamp) =>
      val sv = BSV.zeros[Double](numFeatures)
      sv(userId) = 1.0
      sv(movieId + maxUserId) = 1.0
      sv(timestamp / daySeconds - minTime + maxUserId + maxMovieId) = 1.0
      val gen = (1125899906842597L * timestamp).abs
      val labeledPoint = new LabeledPoint(rating,
        new SSV(sv.length, sv.index.slice(0, sv.used), sv.data.slice(0, sv.used)))
      (gen, labeledPoint)
    }.persist(newLevel)
    dataSet.count()
    movieLens.unpersist()

    val trainSet = dataSet.filter(t => t._1 % 5 > 0).map(_._2).zipWithIndex().map(_.swap).persist(newLevel)
    val testSet = dataSet.filter(t => t._1 % 5 == 0).map(_._2).zipWithIndex().map(_.swap).persist(newLevel)
    trainSet.count()
    testSet.count()
    dataSet.unpersist()

    /**
     * The first view contains [0,maxUserId),The second view contains [maxUserId, maxMovieId + maxUserId)...
     * The third contains [maxMovieId + maxUserId, numFeatures)  The last id equals the number of features
     */
    val views = Array(maxUserId, maxMovieId + maxUserId, numFeatures).map(_.toLong)
    (trainSet, testSet, views)
  }

  def genSamplesSVDPlusPlus(
    sc: SparkContext,
    dataFile: String,
    numPartitions: Int = -1,
    newLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK): (RDD[(Long, LabeledPoint)],
    RDD[(Long, LabeledPoint)], Array[Long]) = {
    val line = sc.textFile(dataFile).first()
    val splitString = if (line.contains(",")) "," else "::"
    var movieLens = sc.textFile(dataFile, sc.defaultParallelism).mapPartitions { iter =>
      iter.filter(t => !t.startsWith("userId") && !t.isEmpty).map { line =>
        val Array(userId, movieId, rating, timestamp) = line.split(splitString)
        (userId.toInt, (movieId.toInt, rating.toDouble))
      }
    }
    movieLens = movieLens.repartition(if (numPartitions > 0) numPartitions else sc.defaultParallelism)
    movieLens.persist(newLevel).count()

    val maxMovieId = movieLens.map(_._2._1).max + 1
    val maxUserId = movieLens.map(_._1).max + 1
    val numFeatures = maxUserId + 2 * maxMovieId
    val dataSet = movieLens.map { case (userId, (movieId, rating)) =>
      val sv = BSV.zeros[Double](maxMovieId)
      sv(movieId) = rating
      (userId, sv)
    }.reduceByKey(_ :+= _).flatMap { case (userId, ratings) =>
      val activeSize = ratings.activeSize
      ratings.activeIterator.map { case (movieId, rating) =>
        val sv = BSV.zeros[Double](numFeatures)
        sv(userId) = 1.0
        sv(movieId + maxUserId) = 1.0
        ratings.activeKeysIterator.foreach { mId =>
          sv(maxMovieId + maxUserId + mId) = 1.0 / math.sqrt(activeSize)
        }
        new LabeledPoint(rating, new SSV(sv.length, sv.index.slice(0, sv.used), sv.data.slice(0, sv.used)))

      }
    }.zipWithIndex().map(_.swap).persist(newLevel)
    dataSet.count()
    movieLens.unpersist()

    val trainSet = dataSet.filter(t => t._2.features.hashCode() % 5 > 0).persist(newLevel)
    val testSet = dataSet.filter(t => t._2.features.hashCode() % 5 == 0).persist(newLevel)
    trainSet.count()
    testSet.count()
    dataSet.unpersist()

    /**
     * The first view contains [0,maxUserId),The second view contains [maxUserId, maxMovieId + maxUserId)...
     * The third contains [maxMovieId + maxUserId, numFeatures)  The last id equals the number of features
     */
    val views = Array(maxUserId, maxMovieId + maxUserId, numFeatures).map(_.toLong)
    (trainSet, testSet, views)
  }
}
