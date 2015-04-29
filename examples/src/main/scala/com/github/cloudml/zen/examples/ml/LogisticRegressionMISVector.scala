package com.github.cloudml.zen.examples.ml

import com.github.cloudml.zen.ml.classification.LogisticRegressionMIS
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils

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

object LogisticRegressionMISVector {
  def main (args: Array[String]) {
    val conf = new SparkConf().setAppName("MIS with Vectors").setMaster("local")
    val sc = new SparkContext(conf)
    val dataSetFile = s"/Users/basin/gitStore/zen/data/binary_classification_data.txt"
    val dataSet = MLUtils.loadLibSVMFile(sc, dataSetFile).map{case LabeledPoint(label, features)=>
      val newLabel = if (label > 0.0) 1.0 else -1.0
      LabeledPoint(newLabel, features)
    }
    val maxIter = 10
    val stepSize = 1
    val lr = new LogisticRegressionMIS(dataSet)
    lr.setStepSize(stepSize)
    var i = 0
    val startedAt = System.currentTimeMillis()
    val (model, lossArr) = lr.run(maxIter)
    println((System.currentTimeMillis() - startedAt) / 1e3)

    lossArr.foreach(println)
  }
}
