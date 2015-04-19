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

import org.apache.spark.Logging
import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.linalg.{Vector => SV}
import org.apache.spark.rdd.RDD

@Experimental
class DBN(val stackedRBM: StackedRBM)
  extends Logging with Serializable {
  lazy val mlp: MLP = {
    val nn = stackedRBM.toMLP()
    val lastLayer = nn.innerLayers(nn.numLayer - 1)
    NNUtil.initUniformDistWeight(lastLayer.weight, 0.01)
    nn.innerLayers(nn.numLayer - 1) = new SoftMaxLayer(lastLayer.weight, lastLayer.bias)
    nn
  }

  def this(topology: Array[Int]) {
    this(new StackedRBM(topology))
  }
}

@Experimental
object DBN extends Logging {
  def train(
    data: RDD[(SV, SV)],
    numIteration: Int,
    topology: Array[Int],
    fraction: Double,
    learningRate: Double,
    weightCost: Double): DBN = {
    val dbn = new DBN(topology)
    pretrain(data, numIteration, dbn, fraction, learningRate, weightCost)
    finetune(data, numIteration, dbn, fraction, learningRate, weightCost)
    dbn
  }

  def pretrain(
    data: RDD[(SV, SV)],
    numIteration: Int,
    dbn: DBN,
    fraction: Double,
    learningRate: Double,
    weightCost: Double): DBN = {
    val stackedRBM = dbn.stackedRBM
    val numLayer = stackedRBM.innerRBMs.length
    StackedRBM.train(data.map(_._1), numIteration, stackedRBM,
      fraction, learningRate, weightCost, numLayer - 1)
    dbn
  }

  def finetune(data: RDD[(SV, SV)],
    numIteration: Int,
    dbn: DBN,
    fraction: Double,
    learningRate: Double,
    weightCost: Double): DBN = {
    MLP.train(data, numIteration, dbn.mlp,
      fraction, learningRate, weightCost)
    dbn
  }
}
