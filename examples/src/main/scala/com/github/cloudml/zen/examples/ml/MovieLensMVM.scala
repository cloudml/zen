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

import com.github.cloudml.zen.ml.recommendation.{MVMModel, MVMRegression}
import com.github.cloudml.zen.ml.util.Logging
import org.apache.spark.graphx2.GraphXUtils
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser

object MovieLensMVM extends Logging {

  case class Params(
    input: String = null,
    out: String = null,
    numIterations: Int = 40,
    numPartitions: Int = -1,
    stepSize: Double = 0.1,
    regular: Double = 0.05,
    rank: Int = 20,
    useAdaGrad: Boolean = false,
    useWeightedLambda: Boolean = false,
    useSVDPlusPlus: Boolean = false,
    kryo: Boolean = true) extends AbstractParams[Params]

  def main(args: Array[String]) {
    val defaultParams = Params()
    val parser = new OptionParser[Params]("MVM") {
      head("MovieLensMVM: an example app for MVM.")
      opt[Int]("numIterations")
        .text(s"number of iterations, default: ${defaultParams.numIterations}")
        .action((x, c) => c.copy(numIterations = x))
      opt[Int]("numPartitions")
        .text(s"number of partitions, default: ${defaultParams.numPartitions}")
        .action((x, c) => c.copy(numPartitions = x))
      opt[Int]("rank")
        .text(s"dim of 3-way interactions, default: ${defaultParams.rank}")
        .action((x, c) => c.copy(rank = x))
      opt[Unit]("kryo")
        .text("use Kryo serialization")
        .action((_, c) => c.copy(kryo = true))
      opt[Double]("stepSize")
        .text(s"stepSize, default: ${defaultParams.stepSize}")
        .action((x, c) => c.copy(stepSize = x))
      opt[Double]("regular")
        .text(s"L2 regularization, default: ${defaultParams.regular}")
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
          | bin/spark-submit --class com.github.cloudml.zen.examples.ml.MovieLensMVM \
          | examples/target/scala-*/zen-examples-*.jar \
          | --rank 20 --numIterations 50 --regular 0.01 --kryo \
          | data/mllib/sample_movielens_data.txt
          | data/mllib/MVM_model
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
    val checkpointDir = s"$out/checkpoint"
    val conf = new SparkConf().setAppName(s"MVM with $params")
    if (kryo) {
      GraphXUtils.registerKryoClasses(conf)
      // conf.set("spark.kryoserializer.buffer.mb", "8")
    }
    val sc = new SparkContext(conf)
    sc.setCheckpointDir(checkpointDir)
    val (trainSet, testSet, views) = if (useSVDPlusPlus) {
      MovieLensUtils.genSamplesSVDPlusPlus(sc, input, numPartitions, storageLevel)
    } else {
      MovieLensUtils.genSamplesWithTime(sc, input, numPartitions, storageLevel)
    }
    val lfm = new MVMRegression(trainSet, stepSize, views, regular, 0.0, rank,
      useAdaGrad, useWeightedLambda, 1, storageLevel)
    var iter = 0
    var model: MVMModel = null
    while (iter < numIterations) {
      val thisItr = math.min(50, numIterations - iter)
      iter += thisItr
      lfm.run(thisItr)
      model = lfm.saveModel()
      model.factors.count()
      val rmse = model.loss(testSet)
      logInfo(f"(Iteration $iter/$numIterations) Test RMSE:                     $rmse%1.4f")
      println(f"(Iteration $iter/$numIterations) Test RMSE:                     $rmse%1.4f")
    }
    model.save(sc, out)
    sc.stop()
  }
}
