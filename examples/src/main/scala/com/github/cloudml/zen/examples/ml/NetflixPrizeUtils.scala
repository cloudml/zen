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

import java.text.SimpleDateFormat
import java.util.{Locale, TimeZone}

import breeze.linalg.{SparseVector => BSV}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{SparseVector => SSV}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable.ArrayBuffer

object NetflixPrizeUtils {

  def genSamplesWithTime(
    sc: SparkContext,
    input: String,
    numPartitions: Int = -1,
    newLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK):
  (RDD[(Long, LabeledPoint)], RDD[(Long, LabeledPoint)], Array[Long]) = {

    val probeFile = s"$input/probe.txt"
    val dataSetFile = s"$input/training_set/*"
    val probe = sc.wholeTextFiles(probeFile).flatMap { case (fileName, txt) =>
      val ab = new ArrayBuffer[(Int, Int)]
      var lastMovieId = -1
      var lastUserId = -1
      txt.split("\n").filter(_.nonEmpty).foreach { line =>
        if (line.endsWith(":")) {
          lastMovieId = line.split(":").head.toInt
        } else {
          lastUserId = line.toInt
          val pair = (lastUserId, lastMovieId)
          ab += pair
        }
      }
      ab.toSeq
    }.collect().toSet

    val simpleDateFormat = new SimpleDateFormat("yyyy-MM-dd", Locale.ROOT)
    simpleDateFormat.setTimeZone(TimeZone.getTimeZone("GMT+08:00"))
    var nfPrize = sc.wholeTextFiles(dataSetFile, sc.defaultParallelism).flatMap { case (fileName, txt) =>
      val Array(movieId, csv) = txt.split(":")
      csv.split("\n").filter(_.nonEmpty).map { line =>
        val Array(userId, rating, timestamp) = line.split(",")
        val day = simpleDateFormat.parse(timestamp).getTime / (1000L * 60 * 60 * 24)
        ((userId.toInt, movieId.toInt), rating.toDouble, day.toInt)
      }
    }
    nfPrize = if (numPartitions > 0) {
      nfPrize.repartition(numPartitions)
    } else {
      nfPrize.repartition(sc.defaultParallelism)
    }
    nfPrize.persist(newLevel).count()

    val maxUserId = nfPrize.map(_._1._1).max + 1
    val maxMovieId = nfPrize.map(_._1._2).max + 1
    val maxTime = nfPrize.map(_._3).max()
    val minTime = nfPrize.map(_._3).min()
    val maxDay = maxTime - minTime + 1
    val numFeatures = maxUserId + maxMovieId + maxDay

    val testSet = nfPrize.mapPartitions { iter =>
      iter.filter(t => probe.contains(t._1)).map {
        case ((userId, movieId), rating, timestamp) =>
          val sv = BSV.zeros[Double](numFeatures)
          sv(userId) = 1.0
          sv(movieId + maxUserId) = 1.0
          sv(timestamp - minTime + maxUserId + maxMovieId) = 1.0
          new LabeledPoint(rating, new SSV(sv.length, sv.index.slice(0, sv.used), sv.data.slice(0, sv.used)))
      }
    }.zipWithIndex().map(_.swap).persist(newLevel)
    testSet.count()

    val trainSet = nfPrize.mapPartitions { iter =>
      iter.filter(t => !probe.contains(t._1)).map {
        case ((userId, movieId), rating, timestamp) =>
          val sv = BSV.zeros[Double](numFeatures)
          sv(userId) = 1.0
          sv(movieId + maxUserId) = 1.0
          sv(timestamp - minTime + maxUserId + maxMovieId) = 1.0
          new LabeledPoint(rating, new SSV(sv.length, sv.index.slice(0, sv.used), sv.data.slice(0, sv.used)))
      }
    }.zipWithIndex().map(_.swap).persist(newLevel)
    trainSet.count()
    nfPrize.unpersist()
    /**
     * The first view contains [0,maxUserId),The second view contains [maxUserId, maxMovieId + maxUserId)...
     * The third contains [maxMovieId + maxUserId, numFeatures)  The last id equals the number of features
     */
    val views = Array(maxUserId, maxMovieId + maxUserId, numFeatures).map(_.toLong)

    (trainSet, testSet, views)

  }
}
