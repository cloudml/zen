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

package com.github.cloudml.zen.examples.ml

import breeze.linalg.{SparseVector => BSV}
import com.github.cloudml.zen.ml.clustering.LDA
import com.github.cloudml.zen.ml.util.SparkUtils._
import org.apache.spark.mllib.linalg.{Vector => SV}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.graphx.GraphXUtils


object LdaDriver {
  def main(args: Array[String]) {
    if (args.length < 10) {
      println("usage: LDATrain <numTopics> <alpha> <beta> <alphaAs> <totalIteration>" +
        " <checkpoint path> <input path> <output path> <sampleRate> <partition num> {<use kryo serialize>}")
      System.exit(1)
    }
    val numTopics = args(0).toInt
    val alpha = args(1).toDouble
    val beta = args(2).toDouble
    val alphaAS = args(3).toDouble
    val totalIter = args(4).toInt

    assert(numTopics > 0)
    assert(alpha > 0)
    assert(beta > 0)
    assert(alphaAS > 0)
    assert(totalIter > 0)

    val appStartedTime = System.currentTimeMillis()
    val checkpointPath = args(5)
    val inputDataPath = args(6)
    val outputRootPath = args(7)
    val sampleRate = args(8).toDouble
    val partitionNum = args(9).toInt
    assert(sampleRate > 0)
    assert(partitionNum > 0)

    val conf = new SparkConf()
    if (args.length > 10){
      conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      conf.set("spark.kryo.registrator", "com.github.cloudml.zen.ml.clustering.LDAKryoRegistrator")
      GraphXUtils.registerKryoClasses(conf)
    }else {
      conf.set("spark.serializer", "org.apache.spark.serializer.JavaSerializer")
    }
    val sc = new SparkContext(conf)
    sc.setCheckpointDir(checkpointPath)

    println("start LDA on user profile")
    println(s"numTopics = $numTopics, totalIteration = $totalIter")
    println(s"alpha = $alpha, beta = $beta, alphaAS = $alphaAS")
    println(s"inputDataPath = $inputDataPath")

    // read data from file
    val trainingDocs = readDocsFromTxt(sc, inputDataPath, sampleRate, partitionNum)
    println(s"trainingDocs count: ${trainingDocs.count()}")
    val trainingTime = runTraining(sc, outputRootPath, numTopics,
      totalIter, alpha, beta, alphaAS, trainingDocs)

    val appEndedTime = System.currentTimeMillis()
    println(s"Training time consumed: $trainingTime seconds")
    println(s" Total time consumed: ${(appEndedTime - appStartedTime) / 1e3} seconds")
    sc.stop()
  }

  def runTraining(sc: SparkContext,
                  outputRootPath: String,
                  numTopics: Int,
                  totalIter: Int,
                  alpha: Double,
                  beta: Double,
                  alphaAS: Double,
                  trainingDocs: RDD[(Long, SV)]): Double = {
    val trainingStartedTime = System.currentTimeMillis()
    val model = LDA.train(trainingDocs, totalIter, numTopics, alpha, beta, alphaAS)
    val trainingEndedTime = System.currentTimeMillis()

    val param = (alpha, beta, alphaAS, totalIter)
    model.save(sc, outputRootPath, isTransposed = true)
    (trainingEndedTime - trainingStartedTime) / 1e3
  }

  def readDocsFromTxt(sc: SparkContext,
                      docsPath: String,
                      sampleRate: Double,
                      partitionNum: Int): RDD[(Long, SV)] = {
    val rawDocs = sc.textFile(docsPath, partitionNum).sample(false, sampleRate)
    convertDocsToBagOfWords(sc, rawDocs)
  }

  def convertDocsToBagOfWords(sc: SparkContext,
                              rawDocs: RDD[String]): RDD[(Long, SV)] = {
    rawDocs.cache()
    val wordsLength = rawDocs.mapPartitions{ iter =>
      val iterator = iter.map { line =>
        val items = line.split("\\t|\\s+")
        var max = Integer.MIN_VALUE
        items.tail.foreach(token => max = math.max(token.split(":")(0).toInt, max))
        max
      }
      Iterator.single[Int](iterator.max)
    }.collect().max + 1
    println(s"the max words id: $wordsLength")
    val result = rawDocs.map { line =>
      val tokens = line.split("\\t|\\s+")
      val docId = tokens(0).toLong
      if (tokens.length == 1) println(tokens.mkString("\t"))
      val docTermCount = BSV.zeros[Double](wordsLength)
      for (t <- tokens.tail) {
        val termCountPair = t.split(':')
        val termId = termCountPair(0).toInt
        val termCount = if (termCountPair.length > 1) {
          termCountPair(1).toDouble
        } else {
          1.0
        }
        docTermCount(termId) += termCount
      }
      if (docTermCount.activeSize < 1) {
        println(s"docTermCount active iterator: ${docTermCount.activeIterator.mkString(";")}")
      }
      (docId, fromBreeze(docTermCount))
    }
    rawDocs.unpersist()
    result
  }

}


