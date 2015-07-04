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

import java.util.{Timer, TimerTask}
import java.lang.ref.WeakReference
import breeze.linalg.{SparseVector => BSV}
import org.apache.spark.{Logging, SparkContext}
import org.apache.spark.mllib.linalg.{SparseVector => SSV}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

private[zen] object MovieLensUtils extends Logging {

  def gcCleaner(delaydSeconds: Int, periodSeconds: Int, tag: String) {
    val timer = new Timer(tag + " cleanup timer", true)
    val task = new TimerTask {
      override def run() {
        try {
          runGC
          logInfo("Ran metadata cleaner for " + tag)
        } catch {
          case e: Exception => logError("Error running cleanup task for " + tag, e)
        }
      }

      /** Run GC and make sure it actually has run */
      def runGC() {
        val weakRef = new WeakReference(new Object())
        val startTime = System.currentTimeMillis
        System.gc() // Make a best effort to run the garbage collection. It *usually* runs GC.
        // Wait until a weak reference object has been GCed
        System.runFinalization()
        while (weakRef.get != null) {
          System.gc()
          System.runFinalization()
          Thread.sleep(200)
          if (System.currentTimeMillis - startTime > 10000) {
            throw new Exception("automatically cleanup error")
          }
        }
      }
    }

    timer.schedule(task, delaydSeconds * 1000, periodSeconds * 1000)
  }

  def genSamplesWithTime(
    sc: SparkContext,
    dataFile: String,
    numPartitions: Int = -1,
    newLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK):
  (RDD[(Long, LabeledPoint)], RDD[(Long, LabeledPoint)], Array[Long]) = {
    val line = sc.textFile(dataFile).first()
    val splitString = if (line.contains(",")) "," else "::"
    var movieLens = sc.textFile(dataFile).mapPartitions { iter =>
      iter.filter(t => !t.startsWith("userId") && !t.isEmpty).map { line =>
        val Array(userId, movieId, rating, timestamp) = line.split(splitString)
        (userId.toInt, movieId.toInt, rating.toDouble, timestamp.toInt)
      }
    }
    if (numPartitions > 0) {
      movieLens = movieLens.repartition(numPartitions)
      movieLens.count()
    }
    movieLens.persist(newLevel)

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

}
