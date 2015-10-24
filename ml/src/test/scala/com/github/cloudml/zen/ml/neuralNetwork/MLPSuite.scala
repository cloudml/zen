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

package com.github.cloudml.zen.ml.neuralNetwork


import com.github.cloudml.zen.ml.util.{Utils, SparkUtils, MnistDatasetSuite}
import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.{Vector => SV}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.scalatest.{FunSuite, Matchers}

class MLPSuite extends FunSuite with MnistDatasetSuite with Matchers {
  test("MLP") {
    val (data, numVisible) = mnistTrainDataset(5000)
    val topology = Array(numVisible, 500, 10)
    val nn = MLP.train(data, 20, 1000, topology, fraction = 0.02,
      learningRate = 0.1, weightCost = 0.0)

    // val nn = MLP.runLBFGS(data, topology, 100, 4000, 1e-5, 0.001)
    // MLP.runSGD(data, nn, 37, 6000, 0.1, 0.5, 0.0)

    val (dataTest, _) = mnistTrainDataset(10000, 5000)
    println("Error: " + MLP.error(dataTest, nn, 100))
  }

  ignore("binary classification") {
    val sparkHome = sys.props.getOrElse("spark.test.home", fail("spark.test.home is not set!"))
    val dataSetFile = s"$sparkHome/data/a5a"
    val checkpoint = s"$sparkHome/target/tmp"
    sc.setCheckpointDir(checkpoint)
    val data = MLUtils.loadLibSVMFile(sc, dataSetFile).map {
      case LabeledPoint(label, features) =>
        val y = BDV.zeros[Double](2)
        y := 0.04 / y.length
        y(if (label > 0) 0 else 1) += 0.96
        (features, SparkUtils.fromBreeze(y))
    }.persist()
    val trainSet = data.filter(_._1.hashCode().abs % 5 == 3).persist()
    val testSet = data.filter(_._1.hashCode().abs % 5 != 3).persist()

    val numVisible = trainSet.first()._1.size
    val topology = Array(numVisible, 30, 2)
    var nn = MLP.train(trainSet, 100, 1000, topology, fraction = 0.02,
      learningRate = 0.05, weightCost = 0.0)

    val modelPath = s"$checkpoint/model"
    nn.save(sc, modelPath)
    nn = MLP.load(sc, modelPath)
    val scoreAndLabels = testSet.map { case (features, label) =>
      val out = nn.predict(SparkUtils.toBreeze(features).toDenseVector.asDenseMatrix.t)
      // Utils.random.nextInt(2).toDouble
      (out(0, 0), if (label(0) > 0.5) 1.0 else 0.0)
    }.persist()
    scoreAndLabels.repartition(1).map(t => s"${t._1}\t${t._2}").
      saveAsTextFile(s"$checkpoint/mlp/${System.currentTimeMillis()}")
    val testAccuracy = new BinaryClassificationMetrics(scoreAndLabels).areaUnderROC()
    println(f"Test AUC = $testAccuracy%1.6f")

  }

}
