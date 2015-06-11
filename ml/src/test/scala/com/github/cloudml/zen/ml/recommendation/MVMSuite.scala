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

import com.github.cloudml.zen.ml.util._
import org.apache.spark.mllib.regression.LabeledPoint
import com.google.common.io.Files
import org.apache.spark.mllib.util.MLUtils
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, sum => brzSum, Vector => BV}
import org.apache.spark.mllib.linalg.{DenseVector => SDV, Vector => SV, SparseVector => SSV}
import org.apache.spark.storage.StorageLevel

import org.scalatest.{Matchers, FunSuite}

class MVMSuite extends FunSuite with SharedSparkContext with Matchers {
  ignore("binary classification") {
    val sparkHome = sys.props.getOrElse("spark.test.home", fail("spark.test.home is not set!"))
    val dataSetFile = s"$sparkHome/data/binary_classification_data.txt"
    val checkpoint = s"$sparkHome/tmp"
    sc.setCheckpointDir(checkpoint)
    val dataSet = MLUtils.loadLibSVMFile(sc, dataSetFile).zipWithIndex().map {
      case (LabeledPoint(label, features), id) =>
        val newLabel = if (label > 0.0) 1.0 else 0.0
        (id, LabeledPoint(newLabel, features))
    }
    val stepSize = 0.1
    val regParam = 1e-2
    val l2 = (regParam, regParam, regParam)
    val rank = 20
    val useAdaGrad = true
    val trainSet = dataSet.cache()
    val fm = new FMClassification(trainSet, stepSize, l2, rank, useAdaGrad)

    val maxIter = 10
    val pps = new Array[Double](maxIter)
    var i = 0
    val startedAt = System.currentTimeMillis()
    while (i < maxIter) {
      fm.run(1)
      val q = fm.forward(i)
      pps(i) = fm.loss(q)
      i += 1
    }
    println((System.currentTimeMillis() - startedAt) / 1e3)
    pps.foreach(println)

    val ppsDiff = pps.init.zip(pps.tail).map { case (lhs, rhs) => lhs - rhs }
    assert(ppsDiff.count(_ > 0).toDouble / ppsDiff.size > 0.05)
    assert(pps.head - pps.last > 0)


    val fmModel = fm.saveModel()
    val tempDir = Files.createTempDir()
    tempDir.deleteOnExit()
    val path = tempDir.toURI.toString
    fmModel.save(sc, path)
    val sameModel = FMModel.load(sc, path)
    assert(sameModel.k === fmModel.k)
    assert(sameModel.classification === fmModel.classification)
    assert(sameModel.factors.sortByKey().map(_._2).collect() ===
      fmModel.factors.sortByKey().map(_._2).collect())
  }

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
    val miniBatchFraction = 1.0
    val Array(trainSet, testSet) = dataSet.randomSplit(Array(0.8, 0.2))
    val numFeatures = trainSet.first()._2.features.size.toLong
    val fm = new MVMRegression(trainSet.cache(), stepSize, Array(numFeatures / 2, numFeatures), regParam,
      rank, useAdaGrad, miniBatchFraction)
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
    val miniBatchFraction = 1
    val Array(trainSet, testSet) = dataSet.randomSplit(Array(0.8, 0.2))
    trainSet.cache()
    testSet.cache()
    val numFeatures = trainSet.first()._2.features.size.toLong
    val fm = new MVMClassification(trainSet, stepSize, Array(20, numFeatures / 2, numFeatures),
      regParam, rank, useAdaGrad, miniBatchFraction)
    fm.run(numIterations)
    val model = fm.saveModel()
    println(f"Test loss: ${model.loss(testSet.cache())}%1.4f")

  }

  ignore("url_combined dataSet") {
    // val dataSetFile = "/input/lbs/recommend/kdda/*"

    import com.github.cloudml.zen.ml.recommendation._
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
    }.zipWithIndex().map(_.swap).persist(StorageLevel.MEMORY_AND_DISK)
    dataSet.count()
    movieLens.unpersist()

    val stepSize = 0.15
    val numIterations = 200
    val regParam = 1e-3
    val l2 = (regParam, regParam, regParam)
    val rank = 40
    val useAdaGrad = true
    val miniBatchFraction = 1
    val Array(trainSet, testSet, _) = dataSet.randomSplit(Array(0.8, 0.1, 0.1))
    trainSet.persist(StorageLevel.MEMORY_AND_DISK).count()
    testSet.persist(StorageLevel.MEMORY_AND_DISK).count()
    dataSet.unpersist()

    //    val fm = new MVMRegression(trainSet, stepSize, Array(maxUserId, maxUserId + maxMovieId, numFeatures),
    //      regParam, rank, useAdaGrad, miniBatchFraction)

    // val fm = new FMRegression(trainSet, stepSize, regParam, regParam, rank, useAdaGrad, miniBatchFraction)

    val fm = new BSFMRegression(trainSet, stepSize, Array(maxUserId, maxUserId + maxMovieId, numFeatures),
      l2, rank, useAdaGrad, miniBatchFraction)

    fm.run(numIterations)
    val model = fm.saveModel()
    println(f"Test loss: ${model.loss(testSet)}%1.4f")

  }

  ignore("Netflix Prize regression") {
    val sparkHome = sys.props.getOrElse("spark.test.home", fail("spark.test.home is not set!"))

    import com.github.cloudml.zen.ml.recommendation._
    val dataSetFile = s"/input/lbs/recommend/toona/nf_prize_dataset/training_set/*"
    val checkpointDir = "/input/lbs/recommend/toona/als/checkpointDir"
    sc.setCheckpointDir(checkpointDir)

    val nfPrize = sc.wholeTextFiles(dataSetFile, 512).flatMap { case (fileName, txt) =>
      val Array(movieId, csv) = txt.split(":")
      csv.split("\n").filter(_.nonEmpty).map { line =>
        val Array(userId, rating, timestamp) = line.split(",")
        (userId.toInt, (movieId.toInt, rating.toDouble))
      }
    }.repartition(512).persist(StorageLevel.MEMORY_AND_DISK)
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
    }.zipWithIndex().map(_.swap).persist(StorageLevel.MEMORY_AND_DISK)
    dataSet.count()
    nfPrize.unpersist()

    val stepSize = 0.1
    val numIterations = 200
    val regParam = 1e-3
    val l2 = (regParam, regParam, regParam)
    val rank = 5
    val useAdaGrad = true
    val miniBatchFraction = 1
    val Array(trainSet, testSet, _) = dataSet.randomSplit(Array(0.8, 0.1, 0.1))
    trainSet.persist(StorageLevel.MEMORY_AND_DISK).count()
    testSet.persist(StorageLevel.MEMORY_AND_DISK).count()
    dataSet.unpersist()

    //  val fm = new MVMRegression(trainSet, stepSize, Array(maxUserId, numFeatures), regParam,
    //   rank, useAdaGrad, miniBatchFraction)

    val fm = new FMRegression(trainSet, stepSize, l2, rank, useAdaGrad, miniBatchFraction)

    fm.run(numIterations)
    val model = fm.saveModel()
    println(f"Test loss: ${model.loss(testSet)}%1.4f")

  }

  test("movieLens 100k regression") {
    val sparkHome = sys.props.getOrElse("spark.test.home", fail("spark.test.home is not set!"))

    import com.github.cloudml.zen.ml.recommendation._
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
    val regParam = 1e-3
    val l2 = (regParam, regParam, regParam)
    val rank = 10
    val useAdaGrad = true
    val miniBatchFraction = 1
    val Array(trainSet, testSet, _) = dataSet.randomSplit(Array(0.7, 0.1, 0.2))
    trainSet.persist(StorageLevel.MEMORY_AND_DISK).count()
    testSet.persist(StorageLevel.MEMORY_AND_DISK).count()

    val fm = new MVMRegression(trainSet, stepSize, Array(maxUserId, numFeatures),
      regParam, rank, useAdaGrad, miniBatchFraction)

    // val fm = new FMRegression(trainSet, stepSize, l2, rank, useAdaGrad,
    // miniBatchFraction, StorageLevel.MEMORY_AND_DISK)

    //  val fm = new BSFMRegression(trainSet, stepSize, Array(maxUserId, numFeatures),
    //   l2, rank, useAdaGrad, miniBatchFraction)

    fm.run(numIterations)
    val model = fm.saveModel()
    println(f"Test loss: ${model.loss(testSet)}%1.4f")

  }
}
