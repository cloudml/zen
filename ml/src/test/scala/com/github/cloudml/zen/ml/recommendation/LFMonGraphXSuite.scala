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

package org.apache.spark.mllib.classification

import org.apache.spark.graphx.VertexId
import org.apache.spark.mllib.classification.LogisticRegressionSuite._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util._
import org.scalatest.{FunSuite, Matchers}

class LFMonGraphXSuite extends FunSuite with LocalClusterSparkContext with Matchers {
  test("10M dataSet") {

    val sparkHome = sys.props.getOrElse("spark.test.home", fail("spark.test.home is not set!"))
    val traindataSetFile = s"${sparkHome}/data/mllib/list_join_action_data_1130"
    val testdataSetFile = s"${sparkHome}/data/mllib/m100r0.validate"
    // val dataSetFile = s"${sparkHome}/data/mllib/kdda.10m.txt"
    // val dataSetFile = s"${sparkHome}/data/mllib/url_combined.10m.txt"
    //val dataSet = sc.parallelize(generateLogisticInput(1.0, 1.0, nPoints = 100, seed = 42), 2)
    //println(dataSet.collect().mkString("\n"))
    //val dataSet = MLUtils.loadLibSVMFile(sc, dataSetFile)
    val traindataSet = sc.textFile(traindataSetFile).map{line =>
      val parts = line.split("\t")
      //LabeledPoint(parts(3).toDouble, Vectors.dense(parts.slice(0, parts.length).map(_.toDouble)))
      LabeledPoint(parts(3).toDouble, Vectors.sparse(2, Array(parts(0).toInt, 100000000 + parts(1).toInt), Array(1.0, 1.0)))
    }


    val validateSet = sc.textFile(testdataSetFile).map{ line =>
      val parts = line.split("\t")
      (parts(0), parts(1), Vectors.sparse(2, Array(parts(0).toInt, 100000000 + parts(1).toInt), Array(1.0, 1.0)))
    }


    //    val dataSetFile = s"/input/lbs/recommend/kdda/*"
    //    val dataSetFile = s"/input/lbs/recommend/url_combined/*"
    //    val dataSetFile = "/input/lbs/recommend/trainingset/*"
    //    val dataSet = MLUtils.loadLibSVMFile(sc, dataSetFile).repartition(72)


    val stepSize = 0.1
    val numIterations = 10
    val regParam = 1e-2
    val rank = 20
    val trainSet = traindataSet.cache()
    val model = LFMonGraphX.train(trainSet, numIterations, stepSize, 0.0, 0.00, rank)
    val result = LFMonGraphX.predict(validateSet, model, rank)
    println(result.count())
    result.map{case (id:VertexId, (user_id, item_id, score)) => user_id + "\t" + item_id + "\t" + score}.saveAsTextFile(s"${sparkHome}/data/mllib/score")
    //result.take(10).foreach{case (id:VertexId, (user_id:String, item_id:String, value: Double)) => println(id + ":" + user_id + ":" + item_id + ":" + value)}
    //result.saveAsTextFile()


    //    val trainSet = dataSet.map(t => {
    //      LabeledPoint(if (t.label > 0) 1 else 0, t.features)
    //    }).cache()
    //    LogisticRegressionWithSGD.train(trainSet, numIterations)


    //    val algorithm = new LogisticRegressionWithLBFGS()
    //    algorithm.optimizer.setNumIterations(1000).setUpdater(new L1Updater()).setRegParam(regParam)
    //    algorithm.run(trainSet)

  }
}
