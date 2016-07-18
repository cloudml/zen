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

package com.github.cloudml.zen.ml.regression

import com.github.cloudml.zen.ml.util._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.scalatest.{Matchers, FunSuite}
import com.github.cloudml.zen.ml.util.SparkUtils._


class LogisticRegressionSuite extends FunSuite with SharedSparkContext with Matchers {

  test("LogisticRegression MIS") {
    val zenHome = sys.props.getOrElse("zen.test.home", fail("zen.test.home is not set!"))
    val dataSetFile = classOf[LogisticRegressionSuite].getClassLoader().getResource("binary_classification_data.txt").toString()
    val dataSet = MLUtils.loadLibSVMFile(sc, dataSetFile)
    val max = dataSet.map(_.features.activeValuesIterator.map(_.abs).sum + 1L).max

    val maxIter = 10
    val stepSize = 1 / (2 * max)
    val trainDataSet = dataSet.zipWithUniqueId().map { case (LabeledPoint(label, features), id) =>
      val newLabel = if (label > 0.0) 1.0 else -1.0
      (id, LabeledPoint(newLabel, features))
    }
    val lr = new LogisticRegressionMIS(trainDataSet, stepSize)
    val pps = new Array[Double](maxIter)
    var i = 0
    val startedAt = System.currentTimeMillis()
    while (i < maxIter) {
      lr.run(1)
      val q = lr.forward(i)
      pps(i) = lr.loss(q)
      i += 1
    }
    println((System.currentTimeMillis() - startedAt) / 1e3)
    pps.foreach(println)

    val ppsDiff = pps.init.zip(pps.tail).map { case (lhs, rhs) => lhs - rhs }
    assert(ppsDiff.count(_ > 0).toDouble / ppsDiff.size > 0.05)
    assert(pps.head - pps.last > 0)
  }

  test("LogisticRegression SGD") {
    val zenHome = sys.props.getOrElse("zen.test.home", fail("zen.test.home is not set!"))
    val dataSetFile = classOf[LogisticRegressionSuite].getClassLoader().getResource("binary_classification_data.txt").toString()
    val dataSet = MLUtils.loadLibSVMFile(sc, dataSetFile)
    val maxIter = 10
    val stepSize = 1
    val trainDataSet = dataSet.zipWithIndex().map { case (LabeledPoint(label, features), id) =>
      val newLabel = if (label > 0.0) 1.0 else 0
      (id, LabeledPoint(newLabel, features))
    }
    val lr = new LogisticRegressionSGD(trainDataSet, stepSize)
    val pps = new Array[Double](maxIter)
    var i = 0
    val startedAt = System.currentTimeMillis()
    while (i < maxIter) {
      lr.run(1)
      val margin = lr.forward(i)
      pps(i) = lr.loss(margin)
      i += 1
    }
    println((System.currentTimeMillis() - startedAt) / 1e3)
    pps.foreach(println)

    val ppsDiff = pps.init.zip(pps.tail).map { case (lhs, rhs) => lhs - rhs }
    assert(ppsDiff.count(_ > 0).toDouble / ppsDiff.size > 0.05)
    assert(pps.head - pps.last > 0)
  }
}
