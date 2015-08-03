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
import com.github.cloudml.zen.ml.util.SparkHacker
import org.apache.hadoop.fs.{InvalidPathException, FileSystem, Path}
import org.apache.spark.deploy.SparkHadoopUtil
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkContext, SparkConf}
// import org.apache.spark.graphx.GraphXUtils


object LDADriver {
  type OptionMap = Map[String, String]

  def main(args: Array[String]) {
    val options = parseArgs(args)
    val appStartedTime = System.currentTimeMillis()

    val numTopics = options("numtopics").toInt
    val alpha = options("alpha").toFloat
    val beta = options("beta").toFloat
    val alphaAS = options("alphaas").toFloat
    val totalIter = options("totaliter").toInt
    val numPartitions = options("numpartitions").toInt
    assert(numTopics > 0)
    assert(alpha > 0F)
    assert(beta > 0F)
    assert(alphaAS > 0F)
    assert(totalIter > 0, "totalIter must be greater than 0")
    assert(numPartitions > 0)

    val inputPath = options("inpath")
    val outputPath = options("outpath")
    val checkpointPath = outputPath + ".checkpoint"

    val sampleRate = options.getOrElse("samplerate", "1.0").toDouble
    assert(sampleRate > 0.0)

    val conf = new SparkConf()
    val LDAAlgorithm = options.getOrElse("ldaalgorithm", "fastlda")
    val storageLevel = StorageLevel.fromString(options.getOrElse("storagelevel", "MEMORY_AND_DISK").toUpperCase)
    val partStrategy = options.getOrElse("partstrategy", "dbh")
    val saveAsSolid = options.getOrElse("saveassolid", "false").toBoolean

//    val useKryoSerializer = options.getOrElse("usekryoserializer", "false").toBoolean
//    if (useKryoSerializer) {
//      conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
//      conf.set("spark.kryo.registrator", "com.github.cloudml.zen.ml.clustering.LDAKryoRegistrator")
//      GraphXUtils.registerKryoClasses(conf)
//    } else {
//      conf.set("spark.serializer", "org.apache.spark.serializer.JavaSerializer")
//    }
    // TODO: Make KryoSerializer work
    conf.set("spark.serializer", "org.apache.spark.serializer.JavaSerializer")

    val fs = FileSystem.get(SparkHadoopUtil.get.newConfiguration(conf))
    if (fs.exists(new Path(outputPath))) {
      throw new InvalidPathException("Output path %s already exists.".format(outputPath))
    }
    fs.delete(new Path(checkpointPath), true)

    val sc = new SparkContext(conf)
    try {
      sc.setCheckpointDir(checkpointPath)

      println("start LDA on user profile")
      println(s"numTopics = $numTopics, totalIteration = $totalIter")
      println(s"alpha = $alpha, beta = $beta, alphaAS = $alphaAS")
      println(s"inputDataPath = $inputPath")

      val trainingDocs = readDocsFromTxt(sc, inputPath, sampleRate, numPartitions, storageLevel)
      val trainingTime = runTraining(sc, outputPath, numTopics, totalIter, alpha, beta, alphaAS,
        trainingDocs, LDAAlgorithm, partStrategy, storageLevel, saveAsSolid)
      println(s"Training time consumed: $trainingTime seconds")

    } finally {
      sc.stop()
      fs.deleteOnExit(new Path(checkpointPath))
      val appEndedTime = System.currentTimeMillis()
      println(s"Total time consumed: ${(appEndedTime - appStartedTime) / 1e3} seconds")
      fs.close()
    }
  }

  def runTraining(sc: SparkContext,
    outputPath: String,
    numTopics: Int,
    totalIter: Int,
    alpha: Float,
    beta: Float,
    alphaAS: Float,
    trainingDocs: RDD[(Long, BSV[Int])],
    LDAAlgorithm: String,
    partStrategy: String,
    storageLevel: StorageLevel,
    saveAsSolid: Boolean): Double = {
    SparkHacker.gcCleaner(15 * 60, 15 * 60, "LDA_gcCleaner")
    val trainingStartedTime = System.currentTimeMillis()
    val termModel = LDA.train(trainingDocs, totalIter, numTopics, alpha, beta, alphaAS,
      LDAAlgorithm, partStrategy, storageLevel)
    val trainingEndedTime = System.currentTimeMillis()

    println("save the model in term-doc view")
    termModel.save(sc, outputPath, isTransposed = true, saveAsSolid)

    (trainingEndedTime - trainingStartedTime) / 1e3
  }

  def readDocsFromTxt(sc: SparkContext,
    docsPath: String,
    sampleRate: Double,
    numPartitions: Int,
    storageLevel: StorageLevel): RDD[(Long, BSV[Int])] = {
    val rawDocs = sc.textFile(docsPath, numPartitions).sample(false, sampleRate).coalesce(numPartitions, true)
    convertDocsToBagOfWords(sc, rawDocs, storageLevel)
  }

  def convertDocsToBagOfWords(sc: SparkContext,
    rawDocs: RDD[String],
    storageLevel: StorageLevel): RDD[(Long, BSV[Int])] = {
    rawDocs.persist(storageLevel).setName("rawDocs")
    val wordsLength = rawDocs.mapPartitions { iter =>
      val iterator = iter.map { line =>
        val items = line.split("\\t|\\s+")
        var max = Integer.MIN_VALUE
        items.tail.foreach(token => max = math.max(token.split(":")(0).toInt, max))
        max
      }
      Iterator.single[Int](iterator.max)
    }.collect().max + 1
    println(s"the max words id: $wordsLength")
    val bowDocs = rawDocs.map { line =>
      val tokens = line.split("\\t|\\s+")
      val docId = tokens(0).toLong
      if (tokens.length == 1) println(tokens.mkString("\t"))
      val docTermCount = BSV.zeros[Int](wordsLength)
      for (t <- tokens.tail) {
        val termCountPair = t.split(':')
        val termId = termCountPair(0).toInt
        val termCount = if (termCountPair.length > 1) {
          termCountPair(1).toInt
        } else {
          1
        }
        docTermCount(termId) += termCount
      }
      if (docTermCount.activeSize < 1) {
        println(s"docTermCount active iterator: ${docTermCount.activeIterator.mkString(";")}")
      }
      (docId, docTermCount)
    }
    bowDocs.persist(storageLevel).setName("bowDocs")
    val numDocs = bowDocs.count()
    println(s"num docs in the corpus: $numDocs")
    rawDocs.unpersist(blocking = false)
    bowDocs
  }

  def parseArgs(args: Array[String]): OptionMap = {
    val usage = "Usage: LDADriver <Args> [Options] <Input path> <Output path>\n" +
      "  Args: -numTopics=<Int> -alpha=<Float> -beta=<Float> -alphaAS=<Float>\n" +
      "        -totalIter=<Int> -numPartitions=<Int>\n" +
      "  Options: -sampleRate=<Double(*1.0)>\n" +
      "           -LDAAlgorithm=<*FastLDA|LightLDA>\n" +
      "           -storageLevel=<StorageLevel(*MEMORY_AND_DISK)>\n" +
      "           -partStrategy=<*DBH|Edge2D>\n" +
      "           -saveAsSolid=<true|*false>"
      // "-useKryoSerializer=<true|*false>"
    if (args.length < 8) {
      println(usage)
      System.exit(1)
    }
    val arglist = args.toList
    def nextOption(map: OptionMap, list: List[String]): OptionMap = {
      def isSwitch(s: String) = s(0) == '-'
      list match {
        case Nil => map
        case head :: Nil if !isSwitch(head) =>
          nextOption(map ++ Map("outpath" -> head), Nil)
        case head :: tail if !isSwitch(head) =>
          nextOption(map ++ Map("inpath" -> head), tail)
        case head :: tail if isSwitch(head) =>
          val kv = head.toLowerCase.split("=", 2)
          nextOption(map ++ Map(kv(0).substring(1) -> kv(1)), tail)
        case _ =>
          println(usage)
          System.exit(1)
          null.asInstanceOf[OptionMap]
      }
    }
    nextOption(Map(), arglist)
  }
}
