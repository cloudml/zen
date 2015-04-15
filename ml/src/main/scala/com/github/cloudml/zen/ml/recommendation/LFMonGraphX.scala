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

import org.apache.spark.graphx._
import org.apache.spark.graphx.impl.GraphImpl
import org.apache.spark.mllib.classification.LFMonGraphX._
import org.apache.spark.mllib.linalg.{DenseVector => SDV}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.Utils
import org.apache.spark.{HashPartitioner, Logging, Partitioner}

import scala.math._

//case class Point(t : Int, data : Array[Double] )

class LFMonGraphX(
  @transient var dataSet: Graph[VD, ED],
  val stepSize: Double,
  val regw: Double,
  val regv: Double,
  val rank : Int,
  @transient var storageLevel: StorageLevel) extends Serializable with Logging {

  def this(
    input: RDD[(VertexId, LabeledPoint)],
    stepSize: Double = 1e-2,
    regw : Double = 1e-2,
    regv : Double = 1e-2,
    rank : Int = 20,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK) {
    this(initializeDataSet(input, rank, storageLevel), stepSize, regw, regv, rank, storageLevel)
  }

  def this(
          input : RDD[(String, String, org.apache.spark.mllib.linalg.Vector)],
          model : VertexRDD[VD],
          rank : Int,
          storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK) {
    this(initializePredictDataSet(input, model, rank, storageLevel), 1e-2, 1e-2, 1e-2, rank, storageLevel)
  }


  if (dataSet.vertices.getStorageLevel == StorageLevel.NONE) {
    dataSet.persist(storageLevel)
  }

  @transient private var innerIter = 1
  @transient private var deltaSum: VertexRDD[Double] = null
  lazy val numFeatures: Long = features.count()
  lazy val numSamples: Long = samples.count()

  def samples: VertexRDD[VD] = {
    dataSet.vertices.filter(t => t._1 < 0)
  }

  def features: VertexRDD[VD] = {
    dataSet.vertices.filter(t => t._1 >= 0)
  }

  // Factorization Machines
  def run(iterations: Int): Unit = {
    for (iter <- 1 to iterations) {
      val previousDataSet = dataSet
      println("start train (Iteration:" + iter + ")")
      logInfo(s"Start train (Iteration $iter/$iterations)")
      val margin = forward()
      margin.setName(s"margin-$iter").persist(storageLevel)
      println(s"train (Iteration $iter/$iterations) cost : ${error(margin)}")
      var gradient = backward(margin)
      gradient.setName(s"gradient-$iter").persist(storageLevel)

      dataSet = updateWeight(gradient, iter)
      dataSet.persist(storageLevel)
      dataSet = checkpoint(dataSet, deltaSum)
      dataSet.vertices.setName(s"vertices-$iter").count()
      dataSet.edges.setName(s"edges-$iter").count()
      previousDataSet.unpersist(blocking = false)
      margin.unpersist(blocking = false)
      gradient.unpersist(blocking = false)
      logInfo(s"End train (Iteration $iter/$iterations)")
      innerIter += 1
    }
  }

  def saveModel(): VertexRDD[VD] = {
//    val numFeatures = features.map(_._1).max().toInt + 1
//    val featureData = new Array[Double](numFeatures)
//    features.toLocalIterator.foreach { case (index, value) =>
//      featureData(index.toInt) = value.apply(0) //output the first weight now
//    }
//    new LogisticRegressionModel(new SDV(featureData), 0.0) //construct this model directly
    features
  }

  private def error(q: VertexRDD[VD]): Double = {
    samples.join(q).map { case (_, (y, m)) =>
//      if (y > 0.0) {
//        y - MLUtils.log1pExp(margin)
//      } else {
//        MLUtils.log1pExp(margin)
//      }
      val pm = logit_predict(m)
      println(pm + ":" + y(0))
      0.5 * (pm - y(0)) * (pm - y(0))
    }.reduce(_ + _) / numSamples
  }

  private def forward_reduce(a : Array[Double], b : Array[Double]): Array[Double] = {
    val result = new Array[Double](a.length)
    for (i <- 0 until a.length) {
      result(i) = a(i) + b(i)
    }
    result
  }
  private def forward(): VertexRDD[Array[Double]] = {
    dataSet.aggregateMessages[Array[Double]](ctx => {
      // val sampleId = ctx.dstId
      // val featureId = ctx.srcId
      //println("fuckrank:" + rank)
      val result = new Array[Double](rank * 2 + 1)
      val x = ctx.attr
      val w = ctx.srcAttr
      result(0) = x * w(0)
      for (i <- 1 to rank) {
        result(i) = x * w(i)
        result(i * 2) = x * x * w(i) * w(i)
      }
      //assert(!z.isNaN)
      ctx.sendToDst(result)
    }, forward_reduce, TripletFields.Src)
  }

  private def predict(arr : Array[Double]): Double = {
    var result : Double = 0
    result += arr(0)
    for (i <- 1 to rank) {
      result += 0.5 * ( arr(i) * arr(i) - arr(2*i))
    }
    result
  }
  private def logit_predict(arr : Array[Double]): Double = {
    var result : Double = 0
    result += arr(0)
    for (i <- 1 to rank) {
      result += 0.5 * ( arr(i) * arr(i) - arr(2*i))
    }
    (1.0 / (1.0 + math.exp(-result)))
  }
  private def sum(arr : Array[Double]) : Double = {
    var result : Double = 0
    for (i <- 1 to rank) {
      result += arr(i)
    }
    result
  }

  private def predict_all(q: VertexRDD[VD]): VertexRDD[Array[Double]] = {
    //println("samples number:" + samples.count())
    //println("predict number:" + q.count())
    samples.innerJoin(q) { (_, y, m) =>
      //Array(sum(m), predict(m) - y(0))
      Array(logit_predict(m))
    }
  }
  private def backward(q: VertexRDD[VD]): VertexRDD[Array[Double]] = {
    //println("samples number:" + samples.count())
    //println("predict number:" + q.count())
    val multiplier = samples.innerJoin(q) { (_, y, m) =>
      //Array(sum(m), predict(m) - y(0))
      Array(sum(m), logit_predict(m) - y(0))
    }
    //println("dataSet edges number:" + dataSet.edges.count())
    //println("dataSet vertex number:" + dataSet.vertices.count())
    //println("feature vertex number:" + features.count())
    val tmp : Graph[VD, ED] = dataSet.outerJoinVertices(multiplier) {(vid, data, deg) =>
      deg.getOrElse(data)
      }
//    val tmp : Graph[VD, ED] = GraphImpl(multiplier, dataSet.edges).outerJoinVertices(features) { (vid, data, deg) =>
//      println("vid:" + vid)
//      deg.getOrElse(data)
//    }
    //println("tmp edges number:" + tmp.edges.count())
    //println("tmp vertex number:" + tmp.vertices.count())
     tmp.aggregateMessages[Array[Double]](ctx => {
      // val sampleId = ctx.dstId
      // val featureId = ctx.srcId
      val x = ctx.attr
      val summAndMulti = ctx.dstAttr
      val sum_m = summAndMulti(0)
      val multi = summAndMulti(1)
      //val grad = x * m
      val m = new Array[Double](rank + 1)
       //println(ctx.srcAttr.mkString(","))
       val vfeatures : Array[Double] = ctx.srcAttr
      m(0) = x * multi
      for (i <- 1 to rank) {
        val grad = sum_m * x - vfeatures(i) * x * x
        m(i) = grad * multi
      }
      ctx.sendToSrc(m) // send the multi directly
    }, forward_reduce, TripletFields.All).mapValues { gradients =>
      gradients.map(_ / numSamples)// / numSamples
    }
  }

  // Updater for L1 regularized problems
  private def updateWeight(delta: VertexRDD[Array[Double]], iter: Int): Graph[VD, ED] = {
    //val thisIterStepSize = if (useAdaGrad) stepSize else stepSize / sqrt(iter)
    val thisIterStepSize = stepSize /// sqrt(iter)
    val thisIterL1StepSize = stepSize / sqrt(iter)
    val newVertices = dataSet.vertices.leftJoin(delta) { (_, attr, gradient) =>
      gradient match {
        case Some(grad) => {
          var weight = attr
          weight(0) -= thisIterStepSize * (grad(0) + regw * weight(0))
          for (i <- 1 to rank) {
            weight(i) -= thisIterStepSize * (grad(i) + regv * weight(i))
          }
          weight
        }
        case None => attr
      }
    }
    GraphImpl(newVertices, dataSet.edges)
  }


  private def checkpoint(corpus: Graph[VD, ED], delta: VertexRDD[Double]): Graph[VD, ED] = {
    if (innerIter % 100 == 0 && corpus.edges.sparkContext.getCheckpointDir.isDefined) {
      logInfo(s"start checkpoint")
      corpus.checkpoint()
      val newVertices = corpus.vertices.mapValues(t => t)
      val newCorpus = GraphImpl(newVertices, corpus.edges)
      newCorpus.checkpoint()
      logInfo(s"end checkpoint")
      if (delta != null) delta.checkpoint()
      newCorpus
    } else {
      corpus
    }
  }
}

object LFMonGraphX {
  private[mllib] type ED = Double
  private[mllib] type VD = Array[Double] //every weight is a double array, arr[0] is w, arr[1] - arr[f] is v1 - vf
  var predict_rdd : RDD[(VertexId, (String, String ))] = null

  def train(
    input: RDD[LabeledPoint],
    numIterations: Int,
    stepSize: Double,
    regw: Double,
    regv: Double,
    rank : Int,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK): VertexRDD[VD] = {
    val data = input.zipWithIndex().map { case (LabeledPoint(label, features), id) =>
      val newLabel = if (label > 0.0) 1.0 else 0.0
      (id, LabeledPoint(newLabel, features)) // from zero
    }
    val lfm = new LFMonGraphX(data, stepSize, regw, regv, rank, storageLevel)
    lfm.run(numIterations)
    val model = lfm.saveModel()
    model
  }

  def predict(
             input: RDD[(String, String, org.apache.spark.mllib.linalg.Vector)],
             model: VertexRDD[VD],
             rank : Int,
             storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK
               ) : VertexRDD[(String, String, Double)] = {
    val lfm = new LFMonGraphX(input, model, rank, storageLevel)
    val margin = lfm.forward

    val result = lfm.predict_all(margin)
    println(result.count() + ":" + predict_rdd.count())
    result.innerJoin(predict_rdd) { case (id, result, features) => {
      (features._1, features._2, result(0))
    }
      /*
    input: RDD[(String, String, org.apache.spark.mllib.linalg.Vector)],
    model: VertexRDD[VD],
    rank : Int,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK
    ) : RDD[(Long, String, String, Double)] = {
      val lfm = new LFMonGraphX(input, model, rank, storageLevel)
      val margin = lfm.forward
      lfm.predict_all(margin).innerJoin(predict_rdd){ case (id, result, features)  => {
        (id, features._1, features._2, result(0))
      }
      */
    }
  }


  private def initializePredictDataSet(
                                 raw_input: RDD[(String, String, org.apache.spark.mllib.linalg.Vector)],
                                 model: VertexRDD[VD],
                                 rank : Int,
                                 storageLevel: StorageLevel): Graph[VD, ED] = {
    val input = raw_input.zipWithIndex().map {
      case((user_id: String, item_id : String, features : org.apache.spark.mllib.linalg.Vector), id) =>
        (id, (user_id, item_id, features))
    }
    predict_rdd = input.map { case (sampleId, (user_id, item_id, features : org.apache.spark.mllib.linalg.Vector)) =>
        val newId = newSampleId(sampleId)
       (newId, (user_id, item_id))
    }
    val edges = input.flatMap { case (sampleId, (user_id, item_id, features : org.apache.spark.mllib.linalg.Vector)) =>
        val newId = newSampleId(sampleId) //sample id
        features.toBreeze.activeIterator.map { case (index, value) =>
          Edge(index, newId, value)
        }
    }

    val vertices = input.map { case (sampleId, (user_id, item_id, features)) =>
      val newId = newSampleId(sampleId)
      (newId, features)
    }
    var dataSet = Graph.fromEdges(edges, null, storageLevel, storageLevel)

    // degree-based hashing  (re-partitioner)
    val numPartitions = edges.partitions.size
    val partitionStrategy = new DBHPartitioner(numPartitions)
    val newEdges = dataSet.outerJoinVertices(dataSet.degrees) { (vid, data, deg) =>
      deg.getOrElse(0)
    }.triplets.mapPartitions { itr =>
      itr.map { e =>
        (partitionStrategy.getPartition(e), Edge(e.srcId, e.dstId, e.attr))
      }
    }.partitionBy(new HashPartitioner(numPartitions)).map(_._2)
    dataSet = Graph.fromEdges(newEdges, null, storageLevel, storageLevel)
    // end degree-based hashing
    // dataSet = dataSet.partitionBy(PartitionStrategy.EdgePartition2D)

    dataSet.outerJoinVertices(model) { (vid, data, deg) =>
      if (vid < 0) {
        //sample point
        Array(0.0)
      }
      else {
        deg match {
          case Some(weights) => {
            weights //feature which has weight
          }
          case None => {
            Array.fill(rank + 1)(0.0) //feature which does not has weight
          }
        }
      }
      //deg.getOrElse(Utils.random.nextGaussian() * 1e-2) //initialize all the weights to gaussion() * 1e-2
      // deg.getOrElse(0)
    }
  }

  private def initializeDataSet(
    input: RDD[(VertexId, LabeledPoint)],
                               rank : Int,
    storageLevel: StorageLevel): Graph[VD, ED] = {
    val edges = input.flatMap { case (sampleId, labelPoint) =>
      val newId = newSampleId(sampleId) //sample id
      labelPoint.features.toBreeze.activeIterator.map { case (index, value) =>
        Edge(index, newId, value)
      }
    }
    val vertices = input.map { case (sampleId, labelPoint) =>
      val newId = newSampleId(sampleId)
      (newId, labelPoint.label)
    }
    var dataSet = Graph.fromEdges(edges, null, storageLevel, storageLevel)

    // degree-based hashing  (re-partitioner)
    val numPartitions = edges.partitions.size
    val partitionStrategy = new DBHPartitioner(numPartitions)
    val newEdges = dataSet.outerJoinVertices(dataSet.degrees) { (vid, data, deg) =>
      deg.getOrElse(0)
    }.triplets.mapPartitions { itr =>
      itr.map { e =>
        (partitionStrategy.getPartition(e), Edge(e.srcId, e.dstId, e.attr))
      }
    }.partitionBy(new HashPartitioner(numPartitions)).map(_._2)
    dataSet = Graph.fromEdges(newEdges, null, storageLevel, storageLevel)
    // end degree-based hashing
    // dataSet = dataSet.partitionBy(PartitionStrategy.EdgePartition2D)

    dataSet.outerJoinVertices(vertices) { (vid, data, deg) =>
      deg match {
        case Some(i) => {
          Array.fill(1)(i) //label point
        }
        case None => {
          val a = Array.fill(rank + 1)(Utils.random.nextGaussian() * 1e-2) //parameter point
          println(a.mkString(","))
          a
        }
      }
      //deg.getOrElse(Utils.random.nextGaussian() * 1e-2) //initialize all the weights to gaussion() * 1e-2
      // deg.getOrElse(0)
    }
  }

  private def newSampleId(id: Long): VertexId = {
    -(id + 1L)
  }
}


/**
 * Degree-Based Hashing, the paper:
 * Distributed Power-law Graph Computing: Theoretical and Empirical Analysis
 */


/*
private class DBHPartitioner(val partitions: Int) extends Partitioner {
  val mixingPrime: Long = 1125899906842597L

  def numPartitions = partitions

  def getPartition(key: Any): Int = {
    val edge = key.asInstanceOf[EdgeTriplet[Int, ED]]
    val srcDeg = edge.srcAttr
    val dstDeg = edge.dstAttr
    val srcId = edge.srcId
    val dstId = edge.dstId
    if (srcDeg < dstDeg) {
      getPartition(srcId)
    } else {
      getPartition(dstId)
    }
  }

  def getPartition(idx: Int): PartitionID = {
    getPartition(idx.toLong)
  }

  def getPartition(idx: Long): PartitionID = {
    (abs(idx * mixingPrime) % partitions).toInt
  }

  override def equals(other: Any): Boolean = other match {
    case h: DBHPartitioner =>
      h.numPartitions == numPartitions
    case _ =>
      false
  }

  override def hashCode: Int = numPartitions
}
*/
