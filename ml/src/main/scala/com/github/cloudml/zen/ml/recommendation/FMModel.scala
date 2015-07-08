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

import com.github.cloudml.zen.ml.recommendation.FM._
import com.github.cloudml.zen.ml.util.LoaderUtils
import com.github.cloudml.zen.ml.util.SparkUtils._
import org.apache.spark.SparkContext
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, RegressionMetrics}
import org.apache.spark.mllib.linalg.{Vector => SV}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.{Loader, Saveable}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.storage.StorageLevel
import org.json4s.DefaultFormats
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

import scala.math._

class FMModel(
  val k: Int,
  val intercept: ED,
  val classification: Boolean,
  val factors: RDD[(Long, VD)]) extends Serializable with Saveable {
  def predict(data: RDD[(Long, SV)]): RDD[(Long, ED)] = {
    data.flatMap { case (sampleId, features) =>
      features.activeIterator.filter(_._2 != 0.0).map {
        case (featureId, value) =>
          (featureId.toLong, (sampleId, value))
      }
    }.join(factors).map { case (featureId, ((sampleId, x), w)) =>
      (sampleId, forwardInterval(k, x, w))
    }.reduceByKey(reduceInterval).map { case (sampleId, arr) =>
      var result = predictInterval(k, intercept, arr)
      if (classification) {
        result = 1.0 / (1.0 + math.exp(-result))
      }
      (sampleId, result)
    }
  }

  def loss(data: RDD[(Long, LabeledPoint)]): Double = {
    // val minTarget = data.map(_._2.label).min()
    // val maxTarget = data.map(_._2.label).max()
    val perd = predict(data.map(t => (t._1, t._2.features)))
    val label = data.map(t => (t._1, t._2.label))
    val scoreAndLabels = label.join(perd).map { case (_, (label, score)) =>
      // var r = Math.max(score, minTarget)
      // r = Math.min(r, maxTarget)
      // pow(l - r, 2)
      (score, label)
    }
    scoreAndLabels.persist(StorageLevel.MEMORY_AND_DISK)
    val ret = if (classification) auc(scoreAndLabels) else rmse(scoreAndLabels)
    scoreAndLabels.unpersist(blocking = false)
    ret
  }

  def rmse(scoreAndLabels: RDD[(Double, Double)]): Double = {
    val metrics = new RegressionMetrics(scoreAndLabels)
    metrics.rootMeanSquaredError
  }

  def auc(scoreAndLabels: RDD[(Double, Double)]): Double = {
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    metrics.areaUnderROC()
  }

  override def save(sc: SparkContext, path: String): Unit = {
    FMModel.SaveLoadV1_0.save(sc, path, k, intercept, classification, factors)
  }

  override protected def formatVersion: String = FMModel.SaveLoadV1_0.formatVersionV1_0
}

object FMModel extends Loader[FMModel] {

  override def load(sc: SparkContext, path: String): FMModel = {
    val (loadedClassName, version, metadata) = LoaderUtils.loadMetadata(sc, path)
    val versionV1_0 = SaveLoadV1_0.formatVersionV1_0
    val classNameV1_0 = SaveLoadV1_0.classNameV1_0
    if (loadedClassName == classNameV1_0 && version == versionV1_0) {
      implicit val formats = DefaultFormats
      val classification = (metadata \ "classification").extract[Boolean]
      val intercept = (metadata \ "intercept").extract[Double]
      val k = (metadata \ "k").extract[Int]
      val dataPath = LoaderUtils.dataPath(path)
      val sqlContext = new SQLContext(sc)
      val dataRDD = sqlContext.parquetFile(dataPath)
      val dataArray = dataRDD.select("featureId", "factors").take(1)
      assert(dataArray.size == 1, s"Unable to load $loadedClassName data from: $dataPath")
      val data = dataArray(0)
      assert(data.size == 2, s"Unable to load $loadedClassName data from: $dataPath")
      val factors = dataRDD.rdd.map {
        case Row(featureId: Long, factors: Seq[Double]) =>
          (featureId, factors.toArray)
      }
      new FMModel(k, intercept, classification, factors)
    } else {
      throw new Exception(
        s"FMModel.load did not recognize model with (className, format version):" +
          s"($loadedClassName, $version).  Supported:\n" +
          s"  ($classNameV1_0, 1.0)")
    }

  }

  private object SaveLoadV1_0 {
    val formatVersionV1_0 = "1.0"
    val classNameV1_0 = "com.github.cloudml.zen.ml.recommendation.FMModel"

    def save(
      sc: SparkContext,
      path: String,
      k: Int,
      intercept: Double,
      classification: Boolean,
      factors: RDD[(Long, Array[Double])]): Unit = {
      val metadata = compact(render
        (("class" -> classNameV1_0) ~ ("version" -> formatVersionV1_0) ~
          ("k" -> k) ~ ("intercept" -> intercept) ~ ("classification" -> classification)))
      sc.parallelize(Seq(metadata), 1).saveAsTextFile(LoaderUtils.metadataPath(path))

      val sqlContext = new SQLContext(sc)
      import sqlContext.implicits._
      // Create Parquet data.
      factors.toDF("featureId", "factors").saveAsParquetFile(LoaderUtils.dataPath(path))
    }
  }

}
