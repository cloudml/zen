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
import com.github.cloudml.zen.ml.recommendation.PartialMVM
import org.apache.spark.graphx.GraphXUtils
import org.apache.spark.mllib.linalg.{SparseVector => SSV}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{Logging, SparkConf, SparkContext}
import scopt.OptionParser

object MovieLensPartialMVM extends Logging {

  case class Params(
    input: String = null,
    out: String = null,
    numIterations: Int = 40,
    numPartitions: Int = -1,
    stepSize: Double = 0.1,
    regular: String = "0.01,0.01,0.01",
    rank: Int = 20,
    useAdaGrad: Boolean = true,
    kryo: Boolean = true) extends AbstractParams[Params]

  def main(args: Array[String]) {
    val defaultParams = Params()
    val parser = new OptionParser[Params]("MovieLensPartialMVM") {
      head("MovieLensPartialMVM: an example app for PartialMVM.")
      opt[Int]("numIterations")
        .text(s"number of iterations, default: ${defaultParams.numIterations}")
        .action((x, c) => c.copy(numIterations = x))
      opt[Int]("numPartitions")
        .text(s"number of partitions, default: ${defaultParams.numPartitions}")
        .action((x, c) => c.copy(numPartitions = x))
      opt[Int]("rank")
        .text(s"dim of 2,3-way interactions, default: ${defaultParams.rank}")
        .action((x, c) => c.copy(rank = x))
      opt[Unit]("kryo")
        .text("use Kryo serialization")
        .action((_, c) => c.copy(kryo = true))
      opt[Double]("stepSize")
        .text(s"stepSize, default: ${defaultParams.stepSize}")
        .action((x, c) => c.copy(stepSize = x))
      opt[String]("regular")
        .text(
          s"""
             |'r0,r1,r2' for SGD: r0=bias regularization,
             |r1=1-way regularization, r2=2-way and 3-way regularization, default: ${defaultParams.regular} (auto)
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
          | bin/spark-submit --class com.github.cloudml.zen.examples.ml.MovieLensPartialMVM \
          | examples/target/scala-*/zen-examples-*.jar \
          | --rank 10 --numIterations 50 --regular 0.01,0.01,0.01 --kryo \
          | data/mllib/sample_movielens_data.txt
          | data/mllib/PartialMVM_model
        """.stripMargin)
    }

    parser.parse(args, defaultParams).map { params =>
      run(params)
    }.getOrElse {
      System.exit(1)
    }
  }

  def run(params: Params): Unit = {
    val Params(input, out, numIterations, numPartitions, stepSize, regular, rank, useAdaGrad, kryo) = params
    val regs = regular.split(",").map(_.toDouble)
    val l2 = (regs(0), regs(1), regs(2))
    val conf = new SparkConf().setAppName(s"PartialMVM with $params")
    if (kryo) {
      GraphXUtils.registerKryoClasses(conf)
      // conf.set("spark.kryoserializer.buffer.mb", "8")
    }
    val sc = new SparkContext(conf)
    val checkpointDir = s"$out/checkpoint"
    sc.setCheckpointDir(checkpointDir)
    val (dataSet, views) = MovieLensUtils.genSamplesWithTime(sc, input, numPartitions)
    val Array(trainSet, testSet) = dataSet.randomSplit(Array(0.8, 0.2))
    trainSet.persist(StorageLevel.MEMORY_AND_DISK).count()
    testSet.persist(StorageLevel.MEMORY_AND_DISK).count()
    dataSet.unpersist()

    val model = PartialMVM.trainRegression(dataSet, numIterations, stepSize, views, l2, rank, useAdaGrad, 1.0)
    model.save(sc, out)
    val rmse = model.loss(testSet)
    logInfo(f"Test RMSE: $rmse%1.4f")
    println(f"Test RMSE: $rmse%1.4f")
    sc.stop()
  }
}
