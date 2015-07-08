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
import com.github.cloudml.zen.ml.util._
import com.google.common.io.Files
import org.apache.spark.mllib.linalg.{DenseVector => SDV, SparseVector => SSV, Vector => SV}
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable.ArrayBuffer

import org.scalatest.{FunSuite, Matchers}

class MVMSuite extends FunSuite with SharedSparkContext with Matchers {

  ignore("regression") {
    val sparkHome = sys.props.getOrElse("spark.test.home", fail("spark.test.home is not set!"))
    val dataSetFile = s"$sparkHome/data/regression_data.txt"
    val checkpointDir = s"$sparkHome/tmp"
    sc.setCheckpointDir(checkpointDir)
    val dataSet = MLUtils.loadLibSVMFile(sc, dataSetFile).zipWithIndex().map {
      case (labeledPoint, id) =>
        (id, labeledPoint)
    }
    val stepSize = 0.01
    val numIterations = 200
    val regParam = 0.0
    val rank = 20
    val useAdaGrad = true
    val useWeightedLambda = true
    val miniBatchFraction = 1.0
    val Array(trainSet, testSet) = dataSet.randomSplit(Array(0.8, 0.2))
    val numFeatures = trainSet.first()._2.features.size.toLong
    val fm = new MVMRegression(trainSet.cache(), stepSize, Array(numFeatures / 2, numFeatures), regParam, regParam,
      rank, useAdaGrad, useWeightedLambda, miniBatchFraction)
    fm.run(numIterations)
    val model = fm.saveModel()
    println(f"Test loss: ${model.loss(testSet.cache())}%1.4f")
  }

  ignore("url_combined classification") {
    val sparkHome = sys.props.getOrElse("spark.test.home", fail("spark.test.home is not set!"))
    val dataSetFile = s"$sparkHome/data/binary_classification_data.txt"
    val checkpointDir = s"$sparkHome/tmp"
    sc.setCheckpointDir(checkpointDir)
    val dataSet = MLUtils.loadLibSVMFile(sc, dataSetFile).zipWithIndex().map {
      case (LabeledPoint(label, features), id) =>
        val newLabel = if (label > 0.0) 1.0 else 0.0
        (id, LabeledPoint(newLabel, features))
    }.cache()
    val stepSize = 0.1
    val numIterations = 500
    val regParam = 1e-3
    val rank = 20
    val useAdaGrad = true
    val useWeightedLambda = true
    val miniBatchFraction = 1
    val Array(trainSet, testSet) = dataSet.randomSplit(Array(0.8, 0.2))
    trainSet.cache()
    testSet.cache()
    val numFeatures = trainSet.first()._2.features.size.toLong
    val fm = new MVMClassification(trainSet, stepSize, Array(20, numFeatures / 2, numFeatures),
      regParam, regParam, rank, useAdaGrad, useWeightedLambda, miniBatchFraction)
    fm.run(numIterations)
    val model = fm.saveModel()
    println(f"Test loss: ${model.loss(testSet.cache())}%1.4f")

  }

  ignore("url_combined dataSet") {
    // val dataSetFile = "/input/lbs/recommend/kdda/*"
    val dataSetFile = "/input/lbs/recommend/url_combined/*"
    val checkpointDir = "/input/lbs/recommend/toona/als/checkpointDir"
    sc.setCheckpointDir(checkpointDir)
    val dataSet = MLUtils.loadLibSVMFile(sc, dataSetFile).zipWithIndex().map {
      case (LabeledPoint(label, features), id) =>
        val newLabel = if (label > 0.0) 1.0 else 0.0
        (id, LabeledPoint(newLabel, features))
    }.repartition(72).persist(StorageLevel.MEMORY_AND_DISK)
    val stepSize = 0.1
    val numIterations = 200
    val regParam = 1e-3
    val l2 = (regParam, regParam, regParam)
    val rank = 20
    val useAdaGrad = true
    val miniBatchFraction = 0.1
    val Array(trainSet, testSet) = dataSet.randomSplit(Array(0.8, 0.2))
    trainSet.persist(StorageLevel.MEMORY_AND_DISK).count()
    testSet.persist(StorageLevel.MEMORY_AND_DISK).count()
    dataSet.unpersist()
    val fm = new FMClassification(trainSet, stepSize, l2, rank, useAdaGrad, miniBatchFraction)
    fm.run(numIterations)
    val model = fm.saveModel()
    println(f"Test loss: ${model.loss(testSet)}%1.4f")

  }

  ignore("movieLens 1m user ID, movie ID, day") {
    val sparkHome = sys.props.getOrElse("spark.test.home", fail("spark.test.home is not set!"))

    import com.github.cloudml.zen.ml.recommendation._
    val dataSetFile = s"/input/lbs/recommend/toona/ml-1m/ratings.dat"
    val checkpointDir = "/input/lbs/recommend/toona/als/checkpointDir"
    sc.setCheckpointDir(checkpointDir)
    val movieLens = sc.textFile(dataSetFile, 2).mapPartitions { iter =>
      iter.filter(t => !t.startsWith("userId") && !t.isEmpty).map { line =>
        val Array(userId, movieId, rating, timestamp) = line.split("::")
        (userId.toInt, movieId.toInt, rating.toDouble, timestamp.toInt / (60 * 60 * 24))
      }
    }.repartition(72).persist(StorageLevel.MEMORY_AND_DISK)
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

    val stepSize = 0.05
    val numIterations = 50
    val regParam = 0.1
    val rank = 20
    val useAdaGrad = true
    val useWeightedLambda = true
    val views = Array(maxUserId, maxUserId + maxMovieId, numFeatures).map(_.toLong)
    val miniBatchFraction = 1
    val Array(trainSet, testSet) = dataSet.randomSplit(Array(0.8, 0.2))
    trainSet.persist(StorageLevel.MEMORY_AND_DISK).count()
    testSet.persist(StorageLevel.MEMORY_AND_DISK).count()

    val fm = new MVMRegression(trainSet, stepSize, views, regParam, 0.0, rank,
      useAdaGrad, useWeightedLambda, miniBatchFraction)
    fm.run(numIterations)
    val model = fm.saveModel()
    println(f"Test loss: ${model.loss(testSet)}%1.4f")


  }

  ignore("movieLens 1m regression") {
    val sparkHome = sys.props.getOrElse("spark.test.home", fail("spark.test.home is not set!"))

    import com.github.cloudml.zen.ml.recommendation._
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

    val stepSize = 0.1
    val numIterations = 50
    val regParam = 0.001
    val l2 = (regParam, regParam, regParam)
    val elasticNetParam = 0.0
    val rank = 20
    val useAdaGrad = true
    val miniBatchFraction = 1
    val views = Array(maxUserId, numFeatures).map(_.toLong)
    val Array(trainSet, testSet) = dataSet.randomSplit(Array(0.8, 0.2))
    trainSet.persist(StorageLevel.MEMORY_AND_DISK).count()
    testSet.persist(StorageLevel.MEMORY_AND_DISK).count()
    dataSet.unpersist()

    //  val fm = new MVMRegression(trainSet, stepSize, views, regParam, elasticNetParam,
    //    rank, useAdaGrad, miniBatchFraction, StorageLevel.MEMORY_AND_DISK)

    val fm = new FMRegression(trainSet, stepSize, l2, rank, useAdaGrad, miniBatchFraction)
    fm.run(numIterations)
    val model = fm.saveModel()
    println(f"Test RMSE: ${model.loss(testSet)}%1.4f")



    //  val fm = new BSFMRegression(trainSet, stepSize, Array(maxUserId, maxUserId + maxMovieId, numFeatures),
    //    l2, rank, useAdaGrad, miniBatchFraction)

  }

  ignore("movieLens 1m ALS") {
    val dataSetFile = s"/input/lbs/recommend/toona/ml-1m/ratings.dat"
    val checkpointDir = "/input/lbs/recommend/toona/als/checkpointDir"
    sc.setCheckpointDir(checkpointDir)

    val movieLens = sc.textFile(dataSetFile, 72).mapPartitions { iter =>
      iter.filter(t => !t.startsWith("userId") && !t.isEmpty).map { line =>
        val Array(userId, movieId, rating, timestamp) = line.split("::")
        Rating(userId.toInt, movieId.toInt, rating.toFloat)
      }
    }.repartition(72).persist(StorageLevel.MEMORY_AND_DISK)
    val maxMovieId = movieLens.map(_.user).max + 1
    val maxUserId = movieLens.map(_.product).max + 1
    val rank = 20
    val numIterations = 200
    val lambda = 1.0
    val Array(trainSet, testSet) = movieLens.randomSplit(Array(0.8, 0.2))
    trainSet.persist(StorageLevel.MEMORY_AND_DISK).count()
    testSet.persist(StorageLevel.MEMORY_AND_DISK).count()
    movieLens.unpersist()

    val model = new ALS().setRank(rank).setIterations(numIterations).setLambda(lambda).run(trainSet)

    val predictions: RDD[Rating] = model.predict(testSet.map(x => (x.user, x.product)))
    val predictionsAndRatings = predictions.map { x =>
      ((x.user, x.product), x.rating)
    }.join(testSet.map(x => ((x.user, x.product), x.rating))).values
    val rmse = math.sqrt(predictionsAndRatings.map(x => (x._1 - x._2) * (x._1 - x._2)).mean())

    println(s"Test RMSE = $rmse.")
  }

  ignore("Netflix Prize time regression") {

    val dataSetFile = s"/input/lbs/recommend/toona/nf_prize_dataset/training_set/*"
    val checkpointDir = "/input/lbs/recommend/toona/als/checkpointDir"
    sc.setCheckpointDir(checkpointDir)

    val nfPrize = sc.wholeTextFiles(dataSetFile, 144).flatMap { case (fileName, txt) =>
      val Array(movieId, csv) = txt.split(":")
      csv.split("\n").filter(_.nonEmpty).map { line =>
        val Array(userId, rating, timestamp) = line.split(",")
        (userId.toInt, (movieId.toInt, rating.toDouble))
      }
    }.repartition(144).persist(StorageLevel.DISK_ONLY)
    val maxMovieId = nfPrize.map(_._2._1).max + 1
    val maxUserId = nfPrize.map(_._1).max + 1
    val numFeatures = maxUserId + maxMovieId
    val dataSet = nfPrize.map { case (userId, (movieId, rating)) =>
      val sv = BSV.zeros[Double](maxMovieId)
      sv(movieId) = rating
      (userId, sv)
    }.reduceByKey(_ :+= _).flatMap { case (userId, ratings) =>
      ratings.activeIterator.map { case (movieId, rating) =>
        val sv = BSV.zeros[Double](numFeatures)
        sv(userId) = 1.0
        sv(movieId + maxUserId) = 1.0
        new LabeledPoint(rating, new SSV(sv.length, sv.index.slice(0, sv.used), sv.data.slice(0, sv.used)))
      }
    }.zipWithIndex().map(_.swap).persist(StorageLevel.DISK_ONLY)
    dataSet.count()
    nfPrize.unpersist()

    val stepSize = 0.1
    val numIterations = 200
    val regParam = 0.1
    val rank = 20
    val useAdaGrad = true
    val useWeightedLambda = true
    val miniBatchFraction = 1
    val views = Array(maxUserId, numFeatures).map(_.toLong)
    val Array(trainSet, testSet) = dataSet.randomSplit(Array(0.8, 0.2))
    trainSet.persist(StorageLevel.DISK_ONLY).count()
    testSet.persist(StorageLevel.DISK_ONLY).count()
    dataSet.unpersist()

    import com.github.cloudml.zen.ml.recommendation._
    val fm = new MVMRegression(trainSet, stepSize, views, regParam, 0.0,
      rank, useAdaGrad, useWeightedLambda, miniBatchFraction, StorageLevel.DISK_ONLY)
    fm.run(numIterations)
    val model = fm.saveModel()
    // val model =  MVMModel.load(sc, "/input/lbs/recommend/toona/nf_prize_dataset/model/")
    println(f"Test RMSE: ${model.loss(testSet)}%1.4f")

  }

  ignore("Netflix Prize regression") {

    val dataSetFile = s"/input/lbs/recommend/toona/nf_prize_dataset/training_set/*"
    val probeFile = "/input/lbs/recommend/toona/nf_prize_dataset/probe.txt"
    val checkpointDir = "/input/lbs/recommend/toona/als/checkpointDir"
    sc.setCheckpointDir(checkpointDir)

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

    val nfPrize = sc.wholeTextFiles(dataSetFile, 144).flatMap { case (fileName, txt) =>
      val Array(movieId, csv) = txt.split(":")
      csv.split("\n").filter(_.nonEmpty).map { line =>
        val Array(userId, rating, timestamp) = line.split(",")
        ((userId.toInt, movieId.toInt), rating.toDouble)
      }
    }.repartition(144).persist(StorageLevel.MEMORY_AND_DISK)

    val maxUserId = nfPrize.map(_._1._1).max + 1
    val maxMovieId = nfPrize.map(_._1._2).max + 1
    val numFeatures = maxUserId + maxMovieId

    val testSet = nfPrize.mapPartitions { iter =>
      iter.filter(t => probe.contains(t._1)).map {
        case ((userId, movieId), rating) =>
          val sv = BSV.zeros[Double](numFeatures)
          sv(userId) = 1.0
          sv(movieId + maxUserId) = 1.0
          new LabeledPoint(rating, new SSV(sv.length, sv.index.slice(0, sv.used), sv.data.slice(0, sv.used)))
      }
    }.zipWithIndex().map(_.swap).persist(StorageLevel.MEMORY_AND_DISK)
    testSet.count()

    val trainSet = nfPrize.mapPartitions { iter =>
      iter.filter(t => !probe.contains(t._1)).map {
        case ((userId, movieId), rating) =>
          val sv = BSV.zeros[Double](numFeatures)
          sv(userId) = 1.0
          sv(movieId + maxUserId) = 1.0
          new LabeledPoint(rating, new SSV(sv.length, sv.index.slice(0, sv.used), sv.data.slice(0, sv.used)))
      }
    }.zipWithIndex().map(_.swap).persist(StorageLevel.MEMORY_AND_DISK)
    trainSet.count()
    nfPrize.unpersist()

    val stepSize = 0.1
    val numIterations = 200
    val regParam = 0.01
    val rank = 24
    val useAdaGrad = true
    val useWeightedLambda = true
    val miniBatchFraction = 1
    val views = Array(maxUserId, numFeatures).map(_.toLong)

    import com.github.cloudml.zen.ml.recommendation._
    val fm = new MVMRegression(trainSet.filter(t => t._1 % 10 > 1), stepSize, views, regParam, 0.0,
      rank, useAdaGrad, useWeightedLambda, miniBatchFraction, StorageLevel.MEMORY_AND_DISK)
    fm.run(numIterations)
    val model = fm.saveModel()
    println(f"Test RMSE: ${model.loss(testSet)}%1.4f")


    // MVMModel.load(sc, "/input/lbs/recommend/toona/nf_prize_dataset/model_mvm/")

  }

  ignore("movieLens 100k regression") {
    val sparkHome = sys.props.getOrElse("spark.test.home", fail("spark.test.home is not set!"))
    val dataSetFile = s"$sparkHome/data/ml-100k/u.data"
    val checkpointDir = s"$sparkHome/tmp"
    sc.setCheckpointDir(checkpointDir)

    val movieLens = sc.textFile(dataSetFile, 2).mapPartitions { iter =>
      iter.filter(t => !t.startsWith("userId") && !t.isEmpty).map { line =>
        val Array(userId, movieId, rating, timestamp) = line.split("\t")
        (userId.toInt, (movieId.toInt, rating.toDouble))
      }
    }.persist(StorageLevel.MEMORY_AND_DISK)
    val maxMovieId = movieLens.map(_._2._1).max + 1
    val maxUserId = movieLens.map(_._1).max + 1
    val numFeatures = maxUserId + maxMovieId
    val dataSet = movieLens.map { case (userId, (movieId, rating)) =>
      val sv = BSV.zeros[Double](maxMovieId)
      sv(movieId) = rating
      (userId, sv)
    }.reduceByKey(_ :+= _).flatMap { case (userId, ratings) =>
      ratings.activeIterator.map { case (movieId, rating) =>
        val sv = BSV.zeros[Double](numFeatures)
        sv(userId) = 1.0
        sv(movieId + maxUserId) = 1.0
        new LabeledPoint(rating, new SSV(sv.length, sv.index.slice(0, sv.used), sv.data.slice(0, sv.used)))
      }
    }.zipWithIndex().map(_.swap).persist(StorageLevel.MEMORY_AND_DISK)
    dataSet.count()
    movieLens.unpersist()

    val stepSize = 0.1
    val numIterations = 200
    val regParam = 0.05
    val l2 = (regParam, regParam, regParam)
    val rank = 10
    val useAdaGrad = true
    val useWeightedLambda = true
    val miniBatchFraction = 1
    val Array(trainSet, testSet) = dataSet.randomSplit(Array(0.8, 0.2))
    trainSet.persist(StorageLevel.MEMORY_AND_DISK).count()
    testSet.persist(StorageLevel.MEMORY_AND_DISK).count()

    val fm = new MVMRegression(trainSet, stepSize, Array(maxUserId, numFeatures),
      regParam, 0.0, rank, useAdaGrad, useWeightedLambda, miniBatchFraction)

    //    val fm = new FMRegression(trainSet, stepSize, l2, rank, useAdaGrad,
    //      miniBatchFraction, StorageLevel.MEMORY_AND_DISK)

    //    val fm = new BSFMRegression(trainSet, stepSize, Array(maxUserId, numFeatures),
    //      l2, rank, useAdaGrad, miniBatchFraction)

    fm.run(numIterations)
    val model = fm.saveModel()
    println(f"Test loss: ${model.loss(testSet)}%1.4f")

  }

  test("movieLens 100k binary classification") {
    val sparkHome = sys.props.getOrElse("spark.test.home", fail("spark.test.home is not set!"))
    val dataSetFile = s"$sparkHome/data/ml-100k/u.data"
    val checkpointDir = s"$sparkHome/tmp"
    sc.setCheckpointDir(checkpointDir)

    val movieLens = sc.textFile(dataSetFile, 2).mapPartitions { iter =>
      iter.filter(t => !t.startsWith("userId") && !t.isEmpty).map { line =>
        val Array(userId, movieId, rating, timestamp) = line.split("\t")
        (userId.toInt, movieId.toInt, rating.toDouble, timestamp.toInt / (60 * 60 * 24))
      }
    }.repartition(72).persist(StorageLevel.MEMORY_AND_DISK)
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
      new LabeledPoint(if (rating > 2.5) 1 else 0, new SSV(sv.length,
        sv.index.slice(0, sv.used), sv.data.slice(0, sv.used)))
    }.zipWithIndex().map(_.swap).persist(StorageLevel.MEMORY_AND_DISK)
    dataSet.count()
    movieLens.unpersist()


    val stepSize = 0.1
    val numIterations = 50
    val regParam = 0.1
    val l2 = (regParam, regParam, regParam)
    val rank = 4
    val useAdaGrad = true
    val useWeightedLambda = true
    val views = Array(maxUserId, maxUserId + maxMovieId, numFeatures).map(_.toLong)
    val miniBatchFraction = 1
    val Array(trainSet, testSet) = dataSet.randomSplit(Array(0.8, 0.2))

    val fm = new FMClassification(trainSet, stepSize, l2,
       rank, useAdaGrad, miniBatchFraction)

    //    val fm = new FMRegression(trainSet, stepSize, l2, rank, useAdaGrad,
    //      miniBatchFraction, StorageLevel.MEMORY_AND_DISK)

    //    val fm = new BSFMRegression(trainSet, stepSize, Array(maxUserId, numFeatures),
    //      l2, rank, useAdaGrad, miniBatchFraction)

    fm.run(numIterations)
    val model = fm.saveModel()
    println(f"Test loss: ${model.loss(testSet)}%1.4f")

  }


  ignore("movieLens 100k SVD++") {
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
    val regParam = 0.01
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
}
