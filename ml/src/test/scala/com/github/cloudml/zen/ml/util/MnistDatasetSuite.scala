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

package com.github.cloudml.zen.ml.util

import breeze.linalg.{DenseVector => BDV}
import org.apache.spark.mllib.linalg.{Vector => SV}
import org.apache.spark.rdd.RDD
import org.scalatest.Suite

import scala.collection.JavaConversions._

trait MnistDatasetSuite extends SharedSparkContext {
  self: Suite =>

  def mnistTrainDataset(size: Int = 5000, dropN: Int = 0): (RDD[(SV, SV)], Int) = {
    val zenHome = sys.props.getOrElse("zen.test.home", fail("spark.test.home is not set!"))
    // http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    val labelsFile = s"$zenHome/data/mnist/train-labels-idx1-ubyte.gz"
    // http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    val imagesFile = s"$zenHome/data/mnist/train-images-idx3-ubyte.gz"
    val minstReader = new MinstDatasetReader(labelsFile, imagesFile)
    val numVisible = minstReader.rows * minstReader.cols
    val minstData = minstReader.slice(dropN, dropN + size).map { case m@MinstItem(label, data) =>
      assert(label < 10)
      val y = BDV.zeros[Double](10)
      y := 0.1 / y.length
      y(label) += 0.9
      val x = m.binaryVector
      (x, SparkUtils.fromBreeze(y))
    }
    val data: RDD[(SV, SV)] = sc.parallelize(minstData.toSeq)
    (data, numVisible)
  }
}
