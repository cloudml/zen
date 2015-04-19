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

import com.github.cloudml.zen.ml.util.MnistDatasetSuite
import org.scalatest.{FunSuite, Matchers}

class MLPSuite extends FunSuite with MnistDatasetSuite with Matchers {
  ignore("MLP") {
    val (data, numVisible) = mnistTrainDataset(5000)
    val topology = Array(numVisible, 500, 10)
    val nn = MLP.train(data, 1000, topology, fraction = 0.02, learningRate = 0.1, weightCost = 0.0)

    // val nn = MLP.runLBFGS(data, topology, 100, 4000, 1e-5, 0.001)
    // MLP.runSGD(data, nn, 37, 6000, 0.1, 0.5, 0.0)

    val (dataTest, _) = mnistTrainDataset(10000, 5000)
    println("Error: " + MLP.error(dataTest, nn, 100))
  }
}
