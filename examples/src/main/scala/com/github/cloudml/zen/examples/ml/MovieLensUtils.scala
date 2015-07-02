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

private[zen] object MovieLensUtils {

  def genSamplesWithTime(
    sc: SparkContext,
    dataFile: String,
    numPartitions: Int = -1,
    newLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK): (RDD[(Long, LabeledPoint)], Array[Long]) = {
    val line = sc.textFile(dataFile).first()
    val splitString = if (line.contains(",")) "," else "::"
    var movieLens = sc.textFile(dataFile).mapPartitions { iter =>
      iter.filter(t => !t.startsWith("userId") && !t.isEmpty).map { line =>
        val Array(userId, movieId, rating, timestamp) = line.split(splitString)
        (userId.toInt, movieId.toInt, rating.toDouble, timestamp.toInt / (60 * 60 * 24))
      }
    }
    if (numPartitions > 0) {
      movieLens = movieLens.repartition(numPartitions)
    }
    movieLens.persist(newLevel)

    val maxUserId = movieLens.map(_._1).max + 1
    val maxMovieId = movieLens.map(_._2).max + 1
    val maxDay = movieLens.map(_._4).max()
    val minDay = movieLens.map(_._4).min()
    val day = maxDay - minDay + 1
    val numFeatures = maxUserId + maxMovieId + day

    val dataSet = movieLens.map { case (userId, movieId, rating, timestamp) =>
      val sv = BSV.zeros[Double](numFeatures)
      sv(userId) = 1.0
      sv(movieId + maxUserId) = 1.0
      sv(timestamp - minDay + maxUserId + maxMovieId) = 1.0
      new LabeledPoint(rating, new SSV(sv.length, sv.index.slice(0, sv.used), sv.data.slice(0, sv.used)))
    }.zipWithIndex().map(_.swap).persist(newLevel)
    dataSet.count()
    movieLens.unpersist()

    /**
     * The first view contains [0,maxUserId),The second view contains [maxUserId, maxMovieId + maxUserId)...
     * The third contains [maxMovieId + maxUserId, numFeatures)  The last id equals the number of features
     */
    val views = Array(maxUserId, maxMovieId + maxUserId, numFeatures).map(_.toLong)
    (dataSet, views)
  }

}
