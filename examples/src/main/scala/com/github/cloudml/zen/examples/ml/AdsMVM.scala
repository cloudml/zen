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
import com.github.cloudml.zen.ml.recommendation._
import com.github.cloudml.zen.ml.util.SparkHacker
import org.apache.spark.graphx.GraphXUtils
import org.apache.spark.mllib.linalg.{SparseVector => SSV}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{Logging, SparkConf, SparkContext}
import scopt.OptionParser

object AdsMVM extends Logging {

  case class Params(
    input: String = null,
    out: String = null,
    numIterations: Int = 200,
    numPartitions: Int = -1,
    stepSize: Double = 0.05,
    regular: Double = 0.01,
    rank: Int = 64,
    useAdaGrad: Boolean = false,
    useWeightedLambda: Boolean = false,
    kryo: Boolean = false) extends AbstractParams[Params]

  def main(args: Array[String]) {
    val defaultParams = Params()
    val parser = new OptionParser[Params]("MVM") {
      head("AdsMVM: an example app for MVM.")
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
      opt[Unit]("weightedLambda")
        .text("use weighted lambda regularization")
        .action((_, c) => c.copy(useWeightedLambda = true))
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
          | bin/spark-submit --class com.github.cloudml.zen.examples.ml.AdsMVM \
          |  examples/target/scala-*/zen-examples-*.jar \
          |  --rank 20 --numIterations 200 --regular 0.01 --kryo \
          |  data/mllib/ads_data/*
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
    val Params(input, out, numIterations, numPartitions, stepSize, regular,
    rank, useAdaGrad, useWeightedLambda, kryo) = params
    val checkpointDir = s"$out/checkpoint"
    val conf = new SparkConf().setAppName(s"MVM with $params")
    if (kryo) {
      GraphXUtils.registerKryoClasses(conf)
      // conf.set("spark.kryoserializer.buffer.mb", "8")
    }
    val sc = new SparkContext(conf)
    sc.setCheckpointDir(checkpointDir)
    SparkHacker.gcCleaner(60 * 15, 60 * 15, "AdsMVM")
    var adaDataSet = sc.textFile(input, sc.defaultParallelism).sample(false, 0.1).map { line =>
      val arr = line.split(" ")
      val Array(label, importance) = arr.head.split(":")
      val features = arr.tail.map { sub =>
        val Array(featureId, featureValue, viewId) = sub.split(":")
        (featureId.toInt, featureValue.toDouble, viewId.toInt)
      }
      (importance.toInt, label.toDouble, features)
    }

    if (numPartitions > 0) adaDataSet = adaDataSet.repartition(numPartitions)
    adaDataSet.persist(StorageLevel.MEMORY_AND_DISK)
    adaDataSet.count()

    val maxUnigramsId = adaDataSet.flatMap(_._3.filter(_._3 == 1).map(_._1)).max
    val maxDisplayUrlId = adaDataSet.flatMap(_._3.filter(_._3 == 2).map(_._1)).max
    val maxPositionId = adaDataSet.flatMap(_._3.filter(_._3 == 3).map(_._1)).max
    val maxMatchTypeId = adaDataSet.flatMap(_._3.filter(_._3 == 4).map(_._1)).max
    val numFeatures = maxMatchTypeId + 1

    val dataSet = adaDataSet.flatMap { case (importance, label, features) =>
      (0 until importance).map { i =>
        val sv = BSV.zeros[Double](numFeatures)
        features.foreach { case (featureId, featureValue, _) =>
          sv(featureId) = featureValue
        }
        new LabeledPoint(label, new SSV(sv.length, sv.index.slice(0, sv.used), sv.data.slice(0, sv.used)))
      }
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
    val fm = new MVMRegression(trainSet, stepSize, views, regular, 0.0, rank,
      useAdaGrad, useWeightedLambda, 1.0, StorageLevel.MEMORY_AND_DISK)
    fm.run(numIterations)
    val model = fm.saveModel()
    model.save(sc, out)
    val rmse = model.loss(testSet)
    logInfo(f"Test RMSE: $rmse%1.4f")
    sc.stop()
    println(f"Test RMSE: $rmse%1.4f")
  }
}
