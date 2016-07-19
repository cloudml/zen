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

import org.scalatest.{Matchers, FunSuite}

class FMSuite extends FunSuite with SharedSparkContext with Matchers {
  test("binary classification") {
    val sparkHome = sys.props.getOrElse("spark.test.home", fail("spark.test.home is not set!"))
    val dataSetFile = classOf[FMSuite].getClassLoader().getResource("binary_classification_data.txt").toString()
    val checkpoint = s"$sparkHome/target/tmp"
    sc.setCheckpointDir(checkpoint)
    val dataSet = MLUtils.loadLibSVMFile(sc, dataSetFile).zipWithIndex().map {
      case (LabeledPoint(label, features), id) =>
        val newLabel = if (label > 0.0) 1.0 else 0.0
        (id, LabeledPoint(newLabel, features))
    }
    val stepSize = 0.1
    val regParam = 1e-4
    val l2 = (regParam, regParam, regParam)
    val rank = 5
    val useAdaGrad = true
    val trainSet = dataSet.cache()
    val fm = new FMClassification(trainSet, stepSize, l2, rank, useAdaGrad)

    val maxIter = 10
    val pps = new Array[Double](maxIter)
    var i = 0
    val startedAt = System.currentTimeMillis()
    while (i < maxIter) {
      fm.run(1)
      pps(i) =  fm.saveModel().loss(trainSet)
      i += 1
    }
    println((System.currentTimeMillis() - startedAt) / 1e3)
    pps.foreach(println)

    val ppsDiff = pps.init.zip(pps.tail).map { case (lhs, rhs) => lhs - rhs }
    assert(ppsDiff.count(_ < 0).toDouble / ppsDiff.size > 0.05)


    val fmModel = fm.saveModel()
    val tempDir = Files.createTempDir()
    tempDir.deleteOnExit()
    val path = tempDir.toURI.toString
    fmModel.save(sc, path)
    val sameModel = FMModel.load(sc, path)
    assert(sameModel.k === fmModel.k)
    assert(sameModel.intercept === fmModel.intercept)
    assert(sameModel.classification === fmModel.classification)
    assert(sameModel.factors.sortByKey().map(_._2).collect() ===
      fmModel.factors.sortByKey().map(_._2).collect())
  }

  ignore("regression") {
    val sparkHome = sys.props.getOrElse("spark.test.home", fail("spark.test.home is not set!"))
    val dataSetFile = classOf[FMSuite].getClassLoader().getResource("regression_data.txt").toString()
    val checkpoint = s"$sparkHome/target/tmp"
    sc.setCheckpointDir(checkpoint)
    val dataSet = MLUtils.loadLibSVMFile(sc, dataSetFile).zipWithIndex().map {
      case (labeledPoint, id) =>
        (id, labeledPoint)
    }
    val stepSize = 0.1
    val numIterations = 200
    val regParam = 1e-3
    val l2 = (regParam, regParam, regParam)
    val rank = 20
    val useAdaGrad = true
    val miniBatchFraction = 1.0
    val Array(trainSet, testSet) = dataSet.randomSplit(Array(0.8, 0.2))
    val fm = new FMRegression(trainSet.cache(), stepSize, l2, rank, useAdaGrad, miniBatchFraction)
    fm.run(numIterations)
    val model = fm.saveModel()
    println(f"Test loss: ${model.loss(testSet.cache())}%1.4f")
  }

  ignore("url_combined dataSet") {
    // val dataSetFile = "/input/lbs/recommend/kdda/*"
    val dataSetFile = "/input/lbs/recommend/url_combined/*"
    val checkpointDir = "/input/lbs/recommend/toona/als/checkpointDir"
    sc.setCheckpointDir(checkpointDir)
    val dataSet = MLUtils.loadLibSVMFile(sc, dataSetFile).zipWithIndex().map {
      case (LabeledPoint(label, features), id) =>
        val newLabel = if (label > 0.0) 1.0 else 0.0
        (id, LabeledPoint(newLabel, features))
    }.repartition(72).cache()
    val stepSize = 0.1
    val numIterations = 500
    val regParam = 0.0
    val l2 = (regParam, regParam, regParam)
    val rank = 20
    val useAdaGrad = true
    val miniBatchFraction = 0.1
    val Array(trainSet, testSet) = dataSet.randomSplit(Array(0.8, 0.2))
    val fm = new FMClassification(trainSet.cache(), stepSize, l2, rank, useAdaGrad, miniBatchFraction)
    fm.run(numIterations)
    val model = fm.saveModel()
    println(f"Test loss: ${model.loss(testSet.cache())}%1.4f")

  }

}
