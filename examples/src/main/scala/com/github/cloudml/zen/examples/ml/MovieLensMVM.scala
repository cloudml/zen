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
import com.github.cloudml.zen.ml.recommendation.MVM
import org.apache.log4j.{Level, Logger}
import org.apache.spark.graphx.GraphXUtils
import org.apache.spark.mllib.linalg.{SparseVector => SSV}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser

object MovieLensMVM {

  case class Params(
    input: String = null,
    out: String = null,
    numIterations: Int = 40,
    stepSize: Double = 0.1,
    regular: Double = 0.05,
    rank: Int = 20,
    useAdaGrad: Boolean = false,
    kryo: Boolean = false) extends AbstractParams[Params]

  def main(args: Array[String]) {
    val defaultParams = Params()
    val parser = new OptionParser[Params]("MVM") {
      head("MovieLensMVM: an example app for MVM.")
      opt[Int]("numIterations")
        .text(s"number of iterations, default: ${defaultParams.numIterations}")
        .action((x, c) => c.copy(numIterations = x))
      opt[Int]("rank")
        .text(s"dim of 2-way interactions, default: ${defaultParams.rank}")
        .action((x, c) => c.copy(rank = x))
      opt[Unit]("kryo")
        .text("use Kryo serialization")
        .action((_, c) => c.copy(kryo = true))
      opt[Double]("stepSize")
        .text(s"stepSize, default: ${defaultParams.stepSize}")
        .action((x, c) => c.copy(stepSize = x))
      opt[Double]("regular")
        .text(
          s"L2 regularization, default: ${defaultParams.regular}".stripMargin)
        .action((x, c) => c.copy(regular = x))
      opt[Unit]("adagrad")
        .text("use AdaGrad")
        .action((_, c) => c.copy(useAdaGrad = true))
      arg[String]("<input>")
        .required()
        .text("input paths")
        .action((x, c) => c.copy(input = x))
      arg[String]("<out>")
        .required()
        .text("out paths (model)")
        .action((x, c) => c.copy(out = x))
      note(
        """
          |For example, the following command runs this app on a synthetic dataset:
          |
          | bin/spark-submit --class com.github.cloudml.zen.examples.ml.MovieLensMVM \
          |  examples/target/scala-*/zen-examples-*.jar \
          |  --rank 10 --numIterations 50 --regular 0.01,0.01,0.01 --kryo \
          |  data/mllib/sample_movielens_data.txt
          |  data/mllib/MVM_model
        """.stripMargin)
    }

    parser.parse(args, defaultParams).map { params =>
      run(params)
    }.getOrElse {
      System.exit(1)
    }
  }

  def run(params: Params): Unit = {
    val Params(input, out, numIterations, stepSize, regular, rank, useAdaGrad, kryo) = params
    val conf = new SparkConf().setAppName(s"MVM with $params")
    if (kryo) {
      GraphXUtils.registerKryoClasses(conf)
      // conf.set("spark.kryoserializer.buffer.mb", "8")
    }
    Logger.getRootLogger.setLevel(Level.WARN)
    val sc = new SparkContext(conf)
    val checkpointDir = s"$out/checkpoint"
    sc.setCheckpointDir(checkpointDir)
    val movieLens = sc.textFile(input).mapPartitions { iter =>
      iter.filter(t => !t.startsWith("userId") && !t.isEmpty).map { line =>
        val Array(userId, movieId, rating, timestamp) = line.split("::")
        (userId.toInt, (movieId.toInt, rating.toDouble))
      }
    }.persist(StorageLevel.MEMORY_AND_DISK)
    val maxMovieId = movieLens.map(_._2._1).max + 1
    val maxUserId = movieLens.map(_._1).max + 1
    val numFeatures = maxUserId + 2 * maxMovieId

    /**
     * The first view contains [0,maxUserId),The second view contains [maxUserId, maxMovieId + maxUserId)...
     * The third contains [maxMovieId + maxUserId,numFeatures)  The last id equals the number of features
     */
    val views = Array(maxUserId, maxMovieId + maxUserId, numFeatures).map(_.toLong)

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
    val Array(trainSet, testSet) = dataSet.randomSplit(Array(0.8, 0.2))
    trainSet.persist(StorageLevel.MEMORY_AND_DISK).count()
    testSet.persist(StorageLevel.MEMORY_AND_DISK).count()
    dataSet.unpersist()
    val model = MVM.trainRegression(trainSet, numIterations, stepSize, views, regular, rank, useAdaGrad, 1.0)
    model.save(sc, out)
    println(f"Test RMSE: ${model.loss(testSet)}%1.4f")
    sc.stop()
  }
}
