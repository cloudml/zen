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

package com.github.cloudml.zen.ml.tree


import java.io.{File, PrintWriter, FileOutputStream}

import org.apache.spark.mllib.tree.model._
import scala.collection.mutable

object treeAggregatorFormat{
  type Lists = (List[String], List[Double], List[Double], List[Int], List[Int], List[Double], List[Double])

  def reformatted(topNode: Node): Lists = {
    val splitFeatures = new mutable.MutableList[String]
    val splitGains = new mutable.MutableList[Double]
    val gainPValues = new mutable.MutableList[Double]
    val lteChildren = new mutable.MutableList[Int]
    val gtChildren = new mutable.MutableList[Int]
    val thresholds = new mutable.MutableList[Double]
    val outputs = new mutable.MutableList[Double]

    var curNonLeafIdx = 0
    var curLeafIdx = 0
    val childIdx = (child: Node) => if (child.isLeaf) {
      curLeafIdx -= 1
      curLeafIdx
    } else {
      curNonLeafIdx += 1
      curNonLeafIdx
    }

    val q = new mutable.Queue[Node]
    q.enqueue(topNode)
    while (q.nonEmpty) {
      val node = q.dequeue()
      if (!node.isLeaf) {
        val split = node.split.get
        val stats = node.stats.get
        splitFeatures += s"I:${split.feature}"
        splitGains += stats.gain
        gainPValues += 0.0
        thresholds += split.threshold
        val left = node.leftNode.get
        val right = node.rightNode.get
        lteChildren += childIdx(left)
        gtChildren += childIdx(right)
        q.enqueue(left)
        q.enqueue(right)
      } else {
        outputs += node.predict.predict
      }
    }
    (splitFeatures.toList, splitGains.toList, gainPValues.toList, lteChildren.toList, gtChildren.toList,
      thresholds.toList, outputs.toList)
  }

  def sequence(path: String, model: DecisionTreeModel, modelId: Int): Unit = {
    val topNode = model.topNode
    val (splitFeatures, splitGains, gainPValues, lteChildren, gtChildren, thresholds, outputs) = reformatted(topNode)
    val numInternalNodes = splitFeatures.length

    val pw = new PrintWriter(new FileOutputStream(new File(path), true))
    pw.write(s"[Evaluator:$modelId]\n")
    pw.write("EvaluatorType=DecisionTree\n")
    pw.write(s"NumInternalNodes=$numInternalNodes\n")

    var str = splitFeatures.mkString("\t")
    pw.write(s"SplitFeatures=$str\n")
    str = splitGains.mkString("\t")
    pw.write(s"SplitGain=$str\n")
    str = gainPValues.mkString("\t")
    pw.write(s"GainPValue=$str\n")
    str = lteChildren.mkString("\t")
    pw.write(s"LTEChild=$str\n")
    str = gtChildren.mkString("\t")
    pw.write(s"GTChild=$str\n")
    str = thresholds.mkString("\t")
    pw.write(s"Threshold=$str\n")
    str = outputs.mkString("\t")
    pw.write(s"Output=$str\n")

    pw.write("\n")
    pw.close()
    println(s"save succeed")
  }

  def appendTreeAggregator(filePath: String,
    index: Int,
    evalNodes: Array[Int],
    evalWeights: Array[Double] = null,
    bias: Double = 0.0,
    Type: String = "Linear"): Unit = {
    val pw = new PrintWriter(new FileOutputStream(new File(filePath), true))

    pw.append(s"[Evaluator:$index]").write("\r\n")
    pw.append(s"EvaluatorType=Aggregator").write("\r\n")

    val numNodes = evalNodes.length
    val defaultWeight = 1.0
    if (evalNodes == null) {
      throw new IllegalArgumentException("there is no evaluators to be aggregated")
    } else {
      pw.append(s"NumNodes=$numNodes").write("\r\n")
      pw.append(s"Nodes=").write("")
      for (eval <- evalNodes) {
        pw.append(s"E:$eval").write("\t")
      }
      pw.write("\r\n")
    }

    var weights = new Array[Double](numNodes)
    if (evalWeights == null) {
      for (i <- 0 until numNodes) {
        weights(i) = defaultWeight
      }
    } else {
      weights = evalWeights
    }

    pw.append(s"Weights=").write("")
    for (weight <- weights) {
      pw.append(s"$weight").write("\t")
    }

    pw.write("\r\n")

    pw.append(s"Bias=$bias").write("\r\n")
    pw.append(s"Type=$Type").write("\r\n")

    pw.close()
  }
}
