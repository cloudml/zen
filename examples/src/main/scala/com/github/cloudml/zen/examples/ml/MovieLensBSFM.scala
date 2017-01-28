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

import com.github.cloudml.zen.ml.recommendation.{BSFMModel, BSFMRegression}
import com.github.cloudml.zen.ml.util.Logging
import org.apache.spark.graphx2.GraphXUtils
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser

object MovieLensBSFM extends Logging {

  case class Params(
    input: String = null,
    out: String = null,
    numIterations: Int = 40,
    numPartitions: Int = -1,
    stepSize: Double = 0.1,
    regular: String = "0.01,0.01,0.01",
    rank: Int = 20,
    useAdaGrad: Boolean = false,
    useWeightedLambda: Boolean = false,
    useSVDPlusPlus: Boolean = false,
    kryo: Boolean = true) extends AbstractParams[Params]

  def main(args: Array[String]) {
    val defaultParams = Params()
    val parser = new OptionParser[Params]("BSFM") {
      head("MovieLensBSFM: an example app for BSFM.")
      opt[Int]("numIterations")
        .text(s"number of iterations, default: ${defaultParams.numIterations}")
        .action((x, c) => c.copy(numIterations = x))
      opt[Int]("rank")
        .text(s"dim of 2-way interactions, default: ${defaultParams.rank}")
        .action((x, c) => c.copy(rank = x))
      opt[Int]("numPartitions")
        .text(s"number of partitions, default: ${defaultParams.numPartitions}")
        .action((x, c) => c.copy(numPartitions = x))
      opt[Unit]("kryo")
        .text("use Kryo serialization")
        .action((_, c) => c.copy(kryo = true))
      opt[Double]("stepSize")
        .text(s"stepSize, default: ${defaultParams.stepSize}")
        .action((x, c) => c.copy(stepSize = x))
      opt[String]("regular")
        .text(
          s"""
             |'r0,r1,r2' for SGD: r0=bias regularization, r1=1-way regularization,
             |r2=2-way regularization, default: ${defaultParams.regular} (auto)
           """.stripMargin)
        .action((x, c) => c.copy(regular = x))
      opt[Unit]("adagrad")
        .text("use AdaGrad")
        .action((_, c) => c.copy(useAdaGrad = true))
      opt[Unit]("weightedLambda")
        .text("use weighted lambda regularization")
        .action((_, c) => c.copy(useWeightedLambda = true))
      opt[Unit]("svdPlusPlus")
        .text("use SVD++")
        .action((_, c) => c.copy(useSVDPlusPlus = true))
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
          | For example, the following command runs this app on a synthetic dataset:
          |
          | bin/spark-submit --class com.github.cloudml.zen.examples.ml.MovieLensBSFM \
          | examples/target/scala-*/zen-examples-*.jar \
          | --rank 10 --numIterations 50 --regular 0.01,0.01,0.01 --kryo \
          | data/mllib/sample_movielens_data.txt
          | data/mllib/bsfm_model
        """.stripMargin)
    }

    parser.parse(args, defaultParams).map { params =>
      run(params)
    }.getOrElse {
      System.exit(1)
    }
  }

  def run(params: Params): Unit = {
    val Params(input, out, numIterations, numPartitions, stepSize, regular, rank,
    useAdaGrad, useWeightedLambda, useSVDPlusPlus, kryo) = params
    val storageLevel = if (useSVDPlusPlus) StorageLevel.DISK_ONLY else StorageLevel.MEMORY_AND_DISK
    val regs = regular.split(",").map(_.toDouble)
    val l2 = (regs(0), regs(1), regs(2))
    val conf = new SparkConf().setAppName(s"MovieLensBSFM with $params")
    if (kryo) {
      GraphXUtils.registerKryoClasses(conf)
      // conf.set("spark.kryoserializer.buffer.mb", "8")
    }
    val sc = new SparkContext(conf)
    val checkpointDir = s"$out/checkpoint"
    sc.setCheckpointDir(checkpointDir)
    val (trainSet, testSet, views) = if (useSVDPlusPlus) {
      MovieLensUtils.genSamplesSVDPlusPlus(sc, input, numPartitions, storageLevel)
    }
    else {
      MovieLensUtils.genSamplesWithTime(sc, input, numPartitions, storageLevel)
    }

    val lfm = new BSFMRegression(trainSet, stepSize, views, l2, rank,
      useAdaGrad, useWeightedLambda, 1.0, storageLevel)
    var iter = 0
    var model: BSFMModel = null
    while (iter < numIterations) {
      val thisItr = math.min(50, numIterations - iter)
      iter += thisItr
      if (model != null) model.factors.unpersist(false)
      lfm.run(thisItr)
      model = lfm.saveModel()
      model.factors.persist(storageLevel)
      model.factors.count()
      val rmse = model.loss(testSet)
      logInfo(f"(Iteration $iter/$numIterations) Test RMSE:                     $rmse%1.4f")
      println(f"(Iteration $iter/$numIterations) Test RMSE:                     $rmse%1.4f")
    }
    model.save(sc, out)
    sc.stop()
  }
}
