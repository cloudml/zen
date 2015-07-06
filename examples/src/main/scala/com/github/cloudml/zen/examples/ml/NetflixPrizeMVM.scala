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
import java.util.{TimeZone, Locale}

import breeze.linalg.{SparseVector => BSV}
import com.github.cloudml.zen.ml.recommendation._
import com.github.cloudml.zen.ml.util.SparkHacker
import org.apache.spark.graphx.GraphXUtils
import org.apache.spark.mllib.linalg.{SparseVector => SSV}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{Logging, SparkConf, SparkContext}
import scopt.OptionParser

import scala.collection.mutable.ArrayBuffer

object NetflixPrizeMVM extends Logging {

  case class Params(
    input: String = null,
    out: String = null,
    numIterations: Int = 200,
    numPartitions: Int = -1,
    stepSize: Double = 0.05,
    regular: Double = 0.01,
    rank: Int = 64,
    useAdaGrad: Boolean = false,
    kryo: Boolean = false) extends AbstractParams[Params]

  def main(args: Array[String]) {
    val defaultParams = Params()
    val parser = new OptionParser[Params]("MVM") {
      head("NetflixPrizeMVM: an example app for MVM.")
      opt[Int]("numIterations")
        .text(s"number of iterations, default: ${defaultParams.numIterations}")
        .action((x, c) => c.copy(numIterations = x))
      opt[Int]("numPartitions")
        .text(s"number of partitions, default: ${defaultParams.numPartitions}")
        .action((x, c) => c.copy(numPartitions = x))
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
          | bin/spark-submit --class com.github.cloudml.zen.examples.ml.NetflixPrizeMVM \
          |  examples/target/scala-*/zen-examples-*.jar \
          |  --rank 20 --numIterations 200 --regular 0.01 --kryo \
          |  data/mllib/nf_prize_dataset
          |  data/mllib/MVM_model
        """.stripMargin)
    }

    parser.parse(args, defaultParams).map { params =>
      run(params)
    } getOrElse {
      System.exit(1)
    }
  }

  def run(params: Params): Unit = {
    val Params(input, out, numIterations, numPartitions, stepSize, regular, rank, useAdaGrad, kryo) = params
    val checkpointDir = s"$out/checkpoint"
    val conf = new SparkConf().setAppName(s"MVM with $params")
    if (kryo) {
      GraphXUtils.registerKryoClasses(conf)
      // conf.set("spark.kryoserializer.buffer.mb", "8")
    }
    val sc = new SparkContext(conf)
    sc.setCheckpointDir(checkpointDir)
    SparkHacker.gcCleaner(60 * 10, 60 * 10, "NetflixPrizeMVM")
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
    if (numPartitions > 0) nfPrize = nfPrize.repartition(numPartitions)
    nfPrize.persist(StorageLevel.MEMORY_AND_DISK)
    nfPrize.count()

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
    }.zipWithIndex().map(_.swap).persist(StorageLevel.MEMORY_AND_DISK)
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
    }.zipWithIndex().map(_.swap).persist(StorageLevel.MEMORY_AND_DISK)
    trainSet.count()
    nfPrize.unpersist()

    /**
     * The first view contains [0,maxUserId),The second view contains [maxUserId,numFeatures)...
     * The last id equals the number of features
     */
    val views = Array(maxUserId, numFeatures).map(_.toLong)
    val fm = new MVMRegression(trainSet, stepSize, views, regular, 0.0, rank,
      useAdaGrad, 1.0, StorageLevel.MEMORY_AND_DISK)
    fm.run(numIterations)
    val model = fm.saveModel()
    model.save(sc, out)
    val rmse = model.loss(testSet)
    logInfo(f"Test RMSE: $rmse%1.4f")
    sc.stop()
    println(f"Test RMSE: $rmse%1.4f")
  }
}
