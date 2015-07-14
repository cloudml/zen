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
import com.github.cloudml.zen.ml.regression.{LogisticRegressionSGD, LogisticRegression}
import com.github.cloudml.zen.ml.util.SparkHacker
import org.apache.spark.graphx.GraphXUtils
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.{SparseVector => SSV}
import org.apache.spark.mllib.regression.{GeneralizedLinearModel, LabeledPoint}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{Logging, SparkConf, SparkContext}
import scopt.OptionParser

object AdsLR extends Logging {

  case class Params(
    input: String = null,
    out: String = null,
    numIterations: Int = 200,
    numPartitions: Int = -1,
    stepSize: Double = 0.05,
    regular: Double = 0.01,
    fraction: Double = 1.0,
    useAdaGrad: Boolean = false,
    useThreeViews: Boolean = false,
    diskOnly: Boolean = false,
    kryo: Boolean = false) extends AbstractParams[Params]

  def main(args: Array[String]) {
    val defaultParams = Params()
    val parser = new OptionParser[Params]("LR") {
      head("AdsLR: an example app for LR.")
      opt[Int]("numIterations")
        .text(s"number of iterations, default: ${defaultParams.numIterations}")
        .action((x, c) => c.copy(numIterations = x))
      opt[Int]("numPartitions")
        .text(s"number of partitions, default: ${defaultParams.numPartitions}")
        .action((x, c) => c.copy(numPartitions = x))
      opt[Unit]("kryo")
        .text("use Kryo serialization")
        .action((_, c) => c.copy(kryo = true))
      opt[Double]("stepSize")
        .text(s"stepSize, default: ${defaultParams.stepSize}")
        .action((x, c) => c.copy(stepSize = x))
      opt[Double]("regular")
        .text(
          s"L2 regularization, default: ${defaultParams.regular}")
        .action((x, c) => c.copy(regular = x))
      opt[Double]("fraction")
        .text(
          s"the sampling fraction, default: ${defaultParams.fraction}")
        .action((x, c) => c.copy(fraction = x))
      opt[Unit]("adagrad")
        .text("use AdaGrad")
        .action((_, c) => c.copy(useAdaGrad = true))
      opt[Unit]("diskOnly")
        .text("use DISK_ONLY storage levels")
        .action((_, c) => c.copy(diskOnly = true))
      opt[Unit]("threeViews")
        .text("use three views")
        .action((_, c) => c.copy(useThreeViews = true))
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
          | bin/spark-submit --class com.github.cloudml.zen.examples.ml.AdsLR \
          |  examples/target/scala-*/zen-examples-*.jar \
          |  --numIterations 200 --regular 0.01 --kryo \
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
    val Params(input, out, numIterations, numPartitions, stepSize, regular, fraction,
    useAdaGrad, useThreeViews, diskOnly, kryo) = params

    val storageLevel = if (diskOnly) StorageLevel.DISK_ONLY else StorageLevel.MEMORY_AND_DISK
    val checkpointDir = s"$out/checkpoint"
    val conf = new SparkConf().setAppName(s"LR with $params")
    if (kryo) {
      GraphXUtils.registerKryoClasses(conf)
      // conf.set("spark.kryoserializer.buffer.mb", "8")
    }
    val sc = new SparkContext(conf)
    sc.setCheckpointDir(checkpointDir)
    SparkHacker.gcCleaner(60 * 15, 60 * 15, "AdsLR")
    val (trainSet, testSet, views) = if (useThreeViews) {
      AdsUtils.genSamplesWithTimeAnd3Views(sc, input, numPartitions, fraction, storageLevel)
    } else {
      AdsUtils.genSamplesWithTime(sc, input, numPartitions, fraction, storageLevel)
    }

    val numFeatures = views.last.toInt
    val lr = new LogisticRegressionSGD(trainSet, stepSize, regular, useAdaGrad, storageLevel)
    var iter = 0
    var model: LogisticRegressionModel = null
    while (iter < numIterations) {
      val thisItr = math.min(50, numIterations - iter)
      iter += thisItr
      lr.run(thisItr)
      model = lr.saveModel(numFeatures).asInstanceOf[LogisticRegressionModel]
      model.clearThreshold()
      val scoreAndLabels = testSet.map { case (_, LabeledPoint(label, features)) =>
        (model.predict(features), label)
      }
      scoreAndLabels.persist(storageLevel)
      scoreAndLabels.count()
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      val auc = metrics.areaUnderROC()
      scoreAndLabels.unpersist(false)
      logInfo(f"(Iteration $iter/$numIterations) Test AUC:                     $auc%1.6f")
      println(f"(Iteration $iter/$numIterations) Test AUC:                     $auc%1.6f")
    }
    model.save(sc, out)
    sc.stop()
  }
}
