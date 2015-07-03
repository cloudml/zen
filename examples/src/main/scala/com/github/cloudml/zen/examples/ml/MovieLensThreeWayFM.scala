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
import com.github.cloudml.zen.ml.recommendation.ThreeWayFM
import org.apache.spark.graphx.GraphXUtils
import org.apache.spark.mllib.linalg.{SparseVector => SSV}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{Logging, SparkConf, SparkContext}
import scopt.OptionParser

object MovieLensThreeWayFM extends Logging {

  case class Params(
    input: String = null,
    out: String = null,
    numIterations: Int = 40,
    numPartitions: Int = -1,
    stepSize: Double = 0.1,
    regular: String = "0.01,0.01,0.01,0.01",
    rank2: Int = 10,
    rank3: Int = 10,
    useAdaGrad: Boolean = true,
    kryo: Boolean = true) extends AbstractParams[Params]

  def main(args: Array[String]) {
    val defaultParams = Params()
    val parser = new OptionParser[Params]("MovieLensThreeWayFM") {
      head("MovieLensThreeWayFM: an example app for ThreeWayFM.")
      opt[Int]("numIterations")
        .text(s"number of iterations, default: ${defaultParams.numIterations}")
        .action((x, c) => c.copy(numIterations = x))
      opt[Int]("numPartitions")
        .text(s"number of partitions, default: ${defaultParams.numPartitions}")
        .action((x, c) => c.copy(numPartitions = x))
      opt[Int]("rank2")
        .text(s"dim of 2-way interactions, default: ${defaultParams.rank2}")
        .action((x, c) => c.copy(rank2 = x))
      opt[Int]("rank3")
        .text(s"dim of 3-way interactions, default: ${defaultParams.rank2}")
        .action((x, c) => c.copy(rank3 = x))
      opt[Unit]("kryo")
        .text("use Kryo serialization")
        .action((_, c) => c.copy(kryo = true))
      opt[Double]("stepSize")
        .text(s"stepSize, default: ${defaultParams.stepSize}")
        .action((x, c) => c.copy(stepSize = x))
      opt[String]("regular")
        .text(
          s"""
             |'r0,r1,r2,r3' for SGD: r0=bias regularization, r1=1-way regularization, r2=2-way regularization,
             |r2=3-way regularization default: ${defaultParams.regular} (auto)
           """.stripMargin)
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
          | For example, the following command runs this app on a synthetic dataset:
          |
          | bin/spark-submit --class com.github.cloudml.zen.examples.ml.MovieLensThreeWayFM \
          | examples/target/scala-*/zen-examples-*.jar \
          | --rank2 10 --rank3 10  --numIterations 50 --regular 0.01,0.01,0.01,0.01 --kryo \
          | data/mllib/sample_movielens_data.txt
          | data/mllib/ThreeWayFM_model
        """.stripMargin)
    }

    parser.parse(args, defaultParams).map { params =>
      run(params)
    }.getOrElse {
      System.exit(1)
    }
  }

  def run(params: Params): Unit = {
    val Params(input, out, numIterations, numPartitions, stepSize, regular, rank2, rank3, useAdaGrad, kryo) = params
    val regs = regular.split(",").map(_.toDouble)
    val l2 = (regs(0), regs(1), regs(2), regs(3))
    val conf = new SparkConf().setAppName(s"ThreeWayFM with $params")
    if (kryo) {
      GraphXUtils.registerKryoClasses(conf)
      // conf.set("spark.kryoserializer.buffer.mb", "8")
    }
    val sc = new SparkContext(conf)
    val checkpointDir = s"$out/checkpoint"
    sc.setCheckpointDir(checkpointDir)
    val (trainSet, testSet, views) = MovieLensUtils.genSamplesWithTime(sc, input, numPartitions)
    val model = ThreeWayFM.trainRegression(trainSet, numIterations, stepSize, views, l2, rank2, rank3, useAdaGrad, 1.0)
    model.save(sc, out)
    val rmse = model.loss(testSet)
    logInfo(f"Test RMSE: $rmse%1.4f")
    println(f"Test RMSE: $rmse%1.4f")
    sc.stop()
  }
}
