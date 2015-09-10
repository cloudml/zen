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

import com.github.cloudml.zen.ml.clustering.LDA
import com.github.cloudml.zen.ml.clustering.LDADefines._
import com.github.cloudml.zen.ml.util.SparkHacker
import org.apache.hadoop.fs.Path
import org.apache.spark.deploy.SparkHadoopUtil
import org.apache.spark.graphx2._
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkContext, SparkConf}


object LDADriver {
  type OptionMap = Map[String, String]

  def main(args: Array[String]) {
    val options = parseArgs(args)
    val appStartedTime = System.currentTimeMillis()

    val numTopics = options("numtopics").toInt
    val alpha = options("alpha").toDouble
    val beta = options("beta").toDouble
    val alphaAS = options("alphaas").toDouble
    val totalIter = options("totaliter").toInt
    val numPartitions = options("numpartitions").toInt
    assert(numTopics > 0, "numTopics must be greater than 0")
    assert(alpha > 0D)
    assert(beta > 0D)
    assert(alphaAS > 0D)
    assert(totalIter > 0, "totalIter must be greater than 0")
    assert(numPartitions > 0, "numPartitions must be greater than 0")

    val inputPath = options("inputpath")
    val outputPath = options("outputpath")
    val checkpointPath = outputPath + ".checkpoint"

    val slvlStr = options.getOrElse("storagelevel", "MEMORY_AND_DISK_SER").toUpperCase
    val storageLevel = StorageLevel.fromString(slvlStr)

    val conf = new SparkConf()
    conf.set(cs_numTopics, s"$numTopics")
    conf.set(cs_numPartitions, s"$numPartitions")
    conf.set(cs_inputPath, inputPath)
    conf.set(cs_outputpath, outputPath)
    conf.set(cs_storageLevel, slvlStr)

    conf.set(cs_sampleRate, options.getOrElse("samplerate", "1.0"))
    conf.set(cs_numThreads, options.getOrElse("numthreads", "1"))
    conf.set(cs_LDAAlgorithm, options.getOrElse("ldaalgorithm", "fastlda"))
    conf.set(cs_accelMethod, options.getOrElse("accelmethod", "alias"))
    conf.set(cs_partStrategy, options.getOrElse("partstrategy", "dbh"))
    conf.set(cs_chkptInterval, options.getOrElse("chkptinterval", "10"))
    conf.set(cs_calcPerplexity, options.getOrElse("calcperplexity", "false"))
    conf.set(cs_saveInterval, options.getOrElse("saveinterval", "0"))
    conf.set(cs_saveAsSolid, options.getOrElse("saveassolid", "false"))

    val useKyro = options.get("usekryo").exists(_.toBoolean)
    if (useKyro) {
      conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      GraphXUtils.registerKryoClasses(conf)
      registerKryoClasses(conf)
    } else {
      conf.set("spark.serializer", "org.apache.spark.serializer.JavaSerializer")
    }

    val execCores = options.get("execcores").map(_.toInt).getOrElse(0)
    if (execCores > 0) {
      conf.set("spark.executor.cores", s"$execCores")
    }

    val hadoopConf = SparkHadoopUtil.get.newConfiguration(conf)
    if (sys.env.contains("HADOOP_CONF_DIR") || sys.env.contains("YARN_CONF_DIR")) {
      val hdfsConfPath = if (sys.env.get("HADOOP_CONF_DIR").isDefined) {
        sys.env.get("HADOOP_CONF_DIR").get + "/core-site.xml"
      } else {
        sys.env.get("YARN_CONF_DIR").get + "/core-site.xml"
      }
      hadoopConf.addResource(new Path(hdfsConfPath))
    }
    val outPath = new Path(outputPath)
    val fs = outPath.getFileSystem(hadoopConf)
    if (fs.exists(outPath)) {
      println(s"Error: output path $outputPath already exists.")
      System.exit(2)
    }
    fs.delete(new Path(checkpointPath), true)

    val sc = new SparkContext(conf)
    try {
      sc.setCheckpointDir(checkpointPath)
      println("start LDA on user profile")
      println(s"numTopics = $numTopics, totalIteration = $totalIter")
      println(s"alpha = $alpha, beta = $beta, alphaAS = $alphaAS")
      println(s"inputDataPath = $inputPath")

      val docs = loadCorpus(sc, storageLevel)
      val trainingTime = runTraining(docs, numTopics, totalIter, alpha, beta, alphaAS, storageLevel)
      println(s"Training time consumed: $trainingTime seconds")

    } finally {
      sc.stop()
      fs.deleteOnExit(new Path(checkpointPath))
      val appEndedTime = System.currentTimeMillis()
      println(s"Total time consumed: ${(appEndedTime - appStartedTime) / 1e3} seconds")
      fs.close()
    }
  }

  def runTraining(docs: EdgeRDD[TA],
    numTopics: Int,
    totalIter: Int,
    alpha: Double,
    beta: Double,
    alphaAS: Double,
    storageLevel: StorageLevel): Double = {
    SparkHacker.gcCleaner(15 * 60, 15 * 60, "LDA_gcCleaner")
    val trainingStartedTime = System.currentTimeMillis()
    val termModel = LDA.train(docs, totalIter, numTopics, alpha, beta, alphaAS, storageLevel)
    val trainingEndedTime = System.currentTimeMillis()
    println("save the model in term-topic view")
    termModel.save(isTransposed=true)
    (trainingEndedTime - trainingStartedTime) / 1e3
  }

  def loadCorpus(sc: SparkContext,
    storageLevel: StorageLevel): EdgeRDD[TA] = {
    val conf = sc.getConf
    val inputPath = conf.get(cs_inputPath)
    val numTopics = conf.get(cs_numTopics).toInt
    val numPartitions = conf.get(cs_numPartitions).toInt
    val sr = conf.get(cs_sampleRate).toDouble
    val rawDocs = sc.textFile(inputPath, numPartitions).sample(false, sr)
    LDA.initializeCorpusEdges(rawDocs, "raw", numTopics, storageLevel)
  }

  def parseArgs(args: Array[String]): OptionMap = {
    val usage = "Usage: LDADriver <Args> [Options] <Input path> <Output path>\n" +
      "  Args: -numTopics=<Int> -alpha=<Double> -beta=<Double> -alphaAS=<Double>\n" +
      "        -totalIter=<Int> -numPartitions=<Int>\n" +
      "  Options: -sampleRate=<Double(*1.0)>\n" +
      "           -numThreads=<Int(*1)>\n" +
      "           -LDAAlgorithm=<*FastLDA|LightLDA>\n" +
      "           -accelMethod=<*Alias|FTree|Hybrid>\n" +
      "           -storageLevel=<StorageLevel(*MEMORY_AND_DISK_SER)>\n" +
      "           -partStrategy=<Edge2D|*DBH|VSDLP|BBR>\n" +
      "           -chkptInterval=<Int(*10)> (0 or negative disables checkpoint)\n" +
      "           -calcPerplexity=<true|*false>\n" +
      "           -saveInterval=<Int(*0)> (0 or negative disables save at intervals)\n" +
      "           -saveAsSolid=<true|*false>\n" +
      "           -execCores=<Int(*0)> (0 uses Spark conf setting)\n" +
      "           -useKryo=<true|*false>"
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
          nextOption(map ++ Map("outputpath" -> head), Nil)
        case head :: tail if !isSwitch(head) =>
          nextOption(map ++ Map("inputpath" -> head), tail)
        case head :: tail if isSwitch(head) =>
          var kv = head.toLowerCase.split("=", 2)
          if (kv.length == 1) {
            kv = head.toLowerCase.split(":", 2)
          }
          if (kv.length == 1) {
            println(s"Error: wrong command line format: $head")
            System.exit(1)
          }
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
