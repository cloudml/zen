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

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, sum => brzSum}
import com.github.cloudml.zen.ml.regression.LinearRegression
import com.github.cloudml.zen.ml.util._
import org.apache.spark.mllib.linalg.{DenseVector => SDV, SparseVector => SSV, Vector => SV}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.storage.StorageLevel
import org.scalatest.{FunSuite, Matchers}

import scala.math._

class HigherOrderFMSuite extends FunSuite with SharedSparkContext with Matchers {

  ignore("movieLens 1m regression") {
    val sparkHome = sys.props.getOrElse("spark.test.home", fail("spark.test.home is not set!"))
    val dataSetFile = s"/input/lbs/recommend/toona/ml-1m/ratings.dat"
    val checkpointDir = "/input/lbs/recommend/toona/als/checkpointDir"
    sc.setCheckpointDir(checkpointDir)

    val movieLens = sc.textFile(dataSetFile, 72).mapPartitions { iter =>
      iter.filter(t => !t.startsWith("userId") && !t.isEmpty).map { line =>
        val Array(userId, movieId, rating, timestamp) = line.split("::")
        (userId.toInt, (movieId.toInt, rating.toDouble))
      }
    }.repartition(72).persist(StorageLevel.MEMORY_AND_DISK)
    val maxMovieId = movieLens.map(_._2._1).max + 1
    val maxUserId = movieLens.map(_._1).max + 1
    val numFeatures = maxUserId + maxMovieId
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
        //  ratings.activeKeysIterator.foreach { mId =>
        //    sv(maxMovieId + maxUserId + mId) = 1.0 / math.sqrt(activeSize)
        //  }
        new LabeledPoint(rating, new SSV(sv.length, sv.index.slice(0, sv.used), sv.data.slice(0, sv.used)))
      }
    }.zipWithIndex().map(_.swap).persist(StorageLevel.MEMORY_AND_DISK)
    dataSet.count()
    movieLens.unpersist()

    val stepSize = 0.05
    val numIterations = 200
    val regParam = 0.1
    val l2 = (regParam, regParam, regParam)
    val elasticNetParam = 0.0
    val rank = 20
    val useAdaGrad = true
    val useWeightedLambda = true
    val miniBatchFraction = 1.0
    val views = Array(maxUserId, numFeatures).map(_.toLong)
    val Array(trainSet, testSet) = dataSet.randomSplit(Array(0.8, 0.2))
    trainSet.persist(StorageLevel.MEMORY_AND_DISK).count()
    testSet.persist(StorageLevel.MEMORY_AND_DISK).count()
    dataSet.unpersist()

    val fm = new MVMRegression(trainSet, stepSize, views, regParam, elasticNetParam,
      rank, useAdaGrad, useWeightedLambda, miniBatchFraction, StorageLevel.MEMORY_AND_DISK)

    fm.run(numIterations)
    val model = fm.saveModel()
    println(f"Test RMSE: ${model.loss(testSet)}%1.4f")

  }

  ignore("movieLens 100k SVD++ regression ") {
    val sparkHome = sys.props.getOrElse("spark.test.home", fail("spark.test.home is not set!"))
    val dataSetFile = s"$sparkHome/data/ml-100k/u.data"
    val checkpointDir = s"$sparkHome/tmp"
    sc.setCheckpointDir(checkpointDir)

    val movieLens = sc.textFile(dataSetFile, 2).mapPartitions { iter =>
      iter.filter(t => !t.startsWith("userId") && !t.isEmpty).map { line =>
        val Array(userId, movieId, rating, timestamp) = line.split("\t")
        (userId.toInt, (movieId.toInt, rating.toDouble))
      }
    }.persist(StorageLevel.DISK_ONLY)
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
    }.zipWithIndex().map(_.swap).persist(StorageLevel.DISK_ONLY)
    dataSet.count()
    movieLens.unpersist()

    val stepSize = 0.1
    val numIterations = 50
    val regParam = 0.0
    val l2 = (regParam, regParam, regParam, regParam)
    val rank = 2
    val useAdaGrad = true
    val useWeightedLambda = true
    val views = Array(maxUserId, maxUserId + maxMovieId, numFeatures).map(_.toLong)
    val miniBatchFraction = 1
    val Array(trainSet, testSet) = dataSet.randomSplit(Array(0.8, 0.2))
    trainSet.persist(StorageLevel.DISK_ONLY).count()
    testSet.persist(StorageLevel.DISK_ONLY).count()

    val fm = new ThreeWayFMRegression(trainSet, stepSize, views, l2, rank, rank,
      useAdaGrad, useWeightedLambda, miniBatchFraction)


    fm.run(numIterations)
    val model = fm.saveModel()
    println(f"Test loss: ${model.loss(testSet)}%1.4f")

  }

  test("movieLens 100k regression ") {
    val sparkHome = sys.props.getOrElse("spark.test.home", fail("spark.test.home is not set!"))
    val dataSetFile = s"$sparkHome/data/ml-100k/u.data"
    val checkpointDir = s"$sparkHome/tmp"
    sc.setCheckpointDir(checkpointDir)

    val movieLens = sc.textFile(dataSetFile, 2).mapPartitions { iter =>
      iter.filter(t => !t.startsWith("userId") && !t.isEmpty).map { line =>
        val Array(userId, movieId, rating, timestamp) = line.split("\t")
        (userId.toInt, movieId.toInt, rating.toDouble, timestamp.toInt / (60 * 60 * 24))
      }
    }.persist(StorageLevel.MEMORY_AND_DISK)
    val maxMovieId = movieLens.map(_._2).max + 1
    val maxUserId = movieLens.map(_._1).max + 1
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
    }.zipWithIndex().map(_.swap).persist(StorageLevel.MEMORY_AND_DISK)
    dataSet.count()
    movieLens.unpersist()

    val stepSize = 0.1
    val numIterations = 50
    val regParam = 1e-2

    val rank = 10
    val useAdaGrad = true
    val views = Array(maxUserId, maxUserId + maxMovieId, numFeatures).map(_.toLong)
    val miniBatchFraction = 1.0
    val Array(trainSet, testSet) = dataSet.randomSplit(Array(0.8, 0.2))
    trainSet.persist(StorageLevel.MEMORY_AND_DISK).count()
    testSet.persist(StorageLevel.MEMORY_AND_DISK).count()

    //    val fm = new HigherOrderBSFMRegression(trainSet, stepSize, views, (regParam, regParam, regParam), rank,
    //      useAdaGrad, miniBatchFraction)

    // val fm = new ThreeWayFMRegression(trainSet, stepSize, views,
    //   (regParam, regParam, regParam, regParam), rank, rank, useAdaGrad, miniBatchFraction)

    val lr = new LinearRegression(trainSet, stepSize, regParam, useAdaGrad, StorageLevel.MEMORY_AND_DISK)
    lr.run(numIterations)
    val model = lr.saveModel()
    val sum = testSet.map { case (_, LabeledPoint(label, features)) =>
      pow(label - model.predict(features), 2)
    }.reduce(_ + _)
    println(f"Test loss: ${sqrt(sum / testSet.count())}%1.4f")


    //    val fm = new MVMRegression(trainSet, stepSize, views, regParam, 0.0, rank, useAdaGrad, miniBatchFraction)
    //    fm.run(numIterations)
    //    val model = fm.saveModel()
    //  println(f"Test loss: ${model.loss(testSet)}%1.4f")

  }
}
