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

import com.github.cloudml.zen.ml.regression.LogisticRegression
import org.apache.log4j.{Level, Logger}
import org.apache.spark.graphx.GraphXUtils
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

import scopt.OptionParser

object BinaryClassification {

  case class Params(
    input: String = null,
    out: String = null,
    numIterations: Int = 200,
    stepSize: Double = 1.0,
    l1: Double = 1e-2,
    epsilon: Double = 1e-4,
    useAdaGrad: Boolean = false,
    kryo: Boolean = false) extends AbstractParams[Params]

  def main(args: Array[String]) {
    val defaultParams = Params()
    val parser = new OptionParser[Params]("BinaryClassification") {
      head("BinaryClassification: an example app for LogisticRegression.")
      opt[Int]("numIterations")
        .text(s"number of iterations, default: ${defaultParams.numIterations}")
        .action((x, c) => c.copy(numIterations = x))
      opt[Double]("epsilon")
        .text(s"epsilon (smoothing constant) for MIS, default: ${defaultParams.epsilon}")
        .action((x, c) => c.copy(epsilon = x))
      opt[Unit]("kryo")
        .text("use Kryo serialization")
        .action((_, c) => c.copy(kryo = true))
      opt[Double]("stepSize")
        .text(s"stepSize, default: ${defaultParams.stepSize}")
        .action((x, c) => c.copy(stepSize = x))
      opt[Double]("l1")
        .text(s"L1 Regularization, default: ${defaultParams.l1} (auto)")
        .action((x, c) => c.copy(l1 = x))
      opt[Unit]("adagrad")
        .text("use AdaGrad")
        .action((_, c) => c.copy(useAdaGrad = true))
      arg[String]("<input>")
        .required()
        .text("input paths (binary labeled data in the LIBSVM format)")
        .action((x, c) => c.copy(input = x))
      arg[String]("<out>")
        .required()
        .text("out paths (model)")
        .action((x, c) => c.copy(out = x))
      note(
        """
          |For example, the following command runs this app on a synthetic dataset:
          |
          | bin/spark-submit --class com.github.cloudml.zen.examples.ml.LogisticRegression \
          |  examples/target/scala-*/zen-examples-*.jar \
          |  --numIterations 200 --lambda 1.0 --kryo \
          |  data/mllib/kdda.txt
          |  data/mllib/lr_model.txt
        """.stripMargin)
    }

    parser.parse(args, defaultParams).map { params =>
      run(params)
    }.getOrElse {
      System.exit(1)
    }
  }

  def run(params: Params): Unit = {
    val Params(input, out, numIterations, stepSize, l1, epsilon, useAdaGrad, useKryo) = params
    val conf = new SparkConf().setAppName(s"LogisticRegression with $params")
    if (useKryo) {
      GraphXUtils.registerKryoClasses(conf)
      // conf.set("spark.kryoserializer.buffer.mb", "8")
    }
    Logger.getRootLogger.setLevel(Level.WARN)
    val sc = new SparkContext(conf)
    val dataSet = MLUtils.loadLibSVMFile(sc, input).zipWithUniqueId().map(_.swap).cache()
    val model = LogisticRegression.trainMIS(dataSet, numIterations, stepSize, l1, epsilon, useAdaGrad)
    model.save(sc, out)
    sc.stop()
  }

}
