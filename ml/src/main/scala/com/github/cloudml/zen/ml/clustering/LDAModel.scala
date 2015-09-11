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

package com.github.cloudml.zen.ml.clustering

import java.io._
import java.lang.ref.SoftReference
import java.util.Random

import LDADefines._
import com.github.cloudml.zen.ml.util._
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, norm, sum}
import com.google.common.base.Charsets
import com.google.common.io.Files
import org.apache.hadoop.io.{NullWritable, Text}
import org.apache.hadoop.mapred.TextOutputFormat
import org.apache.spark.graphx2._
import org.apache.spark.mllib.util.{Loader, Saveable}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.collection.AppendOnlyMap
import org.apache.spark.SparkContext
import org.json4s._
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._


class LocalLDAModel(@transient val termTopicCounters: Array[TC],
  val numTopics: Int,
  val numTerms: Int,
  val numTokens: Long,
  val alpha: Double,
  val beta: Double,
  val alphaAS: Double) extends Serializable {

  @transient val totalTopicCounter = collectTopicCounter()
  @transient val algo = new FastLDA

  private val alphaTRatio = alpha * numTopics / (numTokens - 1 + alphaAS * numTopics)
  private val betaSum = beta * numTerms
  private def itemRatio(topic: Int) = alphaTRatio * (totalTopicCounter(topic) + alphaAS) /
    (totalTopicCounter(topic) + betaSum)

  @transient lazy val wordTableCache = new AppendOnlyMap[Int,
    SoftReference[AliasTable[Double]]](numTerms / 2)
  @transient lazy val tDenseTable = {
    val table = new AliasTable[Double](numTopics)
    algo.tDense(table, itemRatio, beta, numTopics)
  }

  private def collectTopicCounter(): BDV[Count] = {
    val total = BDV.zeros[Count](numTopics)
    termTopicCounters.foreach(ttc => total :++=: ttc)
    total
  }

  /**
   * inference interface
   * @param doc the doc to be inferred
   * @param totalIter overall iterations
   * @param burnIn burn-in iterations
   */
  def inference(
    doc: BSV[Count],
    totalIter: Int = 10,
    burnIn: Int = 5): BV[Double] = {
    require(totalIter > burnIn, "totalIter is less than burnInIter")
    require(totalIter > 0, "totalIter is less than 0")
    require(burnIn > 0, "burnInIter is less than 0")
    val gen = new XORShiftRandom()
    val topicDist = BSV.zeros[Int](numTopics)
    val tokens = vector2Array(doc)
    val topics = new Array[Int](tokens.length)
    var docTopicCounter = uniformDistSampler(gen, tokens, topics, numTopics)
    val docCdf = new CumulativeDist[Double](numTopics)
    for (i <- 1 to totalIter) {
      docTopicCounter = sampleDoc(gen, docTopicCounter, tokens, topics, docCdf)
      if (i > burnIn) topicDist :+= docTopicCounter
    }
    val nm = norm(topicDist, 1)
    topicDist.map(_ / nm)
  }

  private[ml] def vector2Array(bow: BV[Int]): Array[Int] = {
    val docLen = sum(bow)
    val sent = new Array[Int](docLen)
    var offset = 0
    bow.activeIterator.filter(_._2 > 0).foreach { case (term, cn) =>
      for (i <- 0 until cn) {
        sent(offset) = term
        offset += 1
      }
    }
    sent
  }

  private[ml] def sampleDoc(gen: Random,
    docTopicCounter: TC,
    tokens: Array[Int],
    topics: Array[Int],
    docCdf: CumulativeDist[Double]): TC = {
    for (i <- topics.indices) {
      val termId = tokens(i)
      val termTopicCounter = termTopicCounters(termId)
      val currentTopic = topics(i)
      algo.dSparse(docCdf, totalTopicCounter, termTopicCounter, docTopicCounter, beta, betaSum)
      val wSparseTable = wordTable(wordTableCache, totalTopicCounter, termTopicCounter, termId)
      val newTopic = algo.tokenSampling(gen, tDenseTable, wSparseTable, docCdf, termTopicCounter,
        docTopicCounter, currentTopic)
      if (newTopic != currentTopic) {
        topics(i) = newTopic
        docTopicCounter(newTopic) += 1
        docTopicCounter(currentTopic) -= 1
      }
    }
    // docTopicCounter.compact()
    docTopicCounter
  }

  private[ml] def wordTable(
    cacheMap: AppendOnlyMap[Int, SoftReference[AliasTable[Double]]],
    totalTopicCounter: BDV[Count],
    termTopicCounter: TC,
    termId: Int): AliasTable[Double] = {
    if (termTopicCounter.used == 0) return null
    var w = cacheMap(termId)
    if (w == null || w.get() == null) {
      val table = new AliasTable[Double](termTopicCounter.used)
      algo.wSparse(table, itemRatio, termTopicCounter)
      w = new SoftReference(table)
      cacheMap.update(termId, w)
    }
    w.get()
  }

  def save(path: String): Unit = {
    val file = new File(path)
    require(!file.exists, s"model file $path does exist")
    Files.touch(file)
    val fw = Files.newWriter(file, Charsets.UTF_8)
    fw.write(s"$numTopics $numTerms $numTokens $alpha $beta $alphaAS \n")
    termTopicCounters.zipWithIndex.foreach { case (sv, index) =>
      val line = s"$index ${sv.activeIterator.filter(_._2 != 0).map(t => s"${t._1}:${t._2}").mkString(" ")}\n"
      fw.write(line)
    }
    fw.close()
  }
}

class DistributedLDAModel(@transient val termTopicCounters: RDD[(VertexId, TC)],
  val numTopics: Int,
  val numTerms: Int,
  val numTokens: Long,
  val alpha: Double,
  val beta: Double,
  val alphaAS: Double,
  var storageLevel: StorageLevel) extends Serializable with Saveable {

  @transient val totalTopicCounter = termTopicCounters.map(_._2)
    .aggregate(BDV.zeros[Count](numTopics))(_ :+= _, _ :+= _)
  @transient val algo = new FastLDA

  /**
   * inference interface
   * @param bowDocs   tuple pair: (dicId, Vector), in which 'docId' is unique
   *                  recommended storage level: StorageLevel.MEMORY_AND_DISK
   * @param totalIter overall iterations
   * @param burnIn    previous burnIn iters results will discard
   */
  def inference(bowDocs: RDD[BOW],
    totalIter: Int = 25,
    burnIn: Int = 22): RDD[(VertexId, HashVector[Double])] = {
    require(totalIter > burnIn, "totalIter is less than burnInIter")
    require(totalIter > 0, "totalIter is less than 0")
    require(burnIn > 0, "burnIn is less than 0")
    val docs = LDA.initializeCorpusEdges(bowDocs, "bow", numTopics, storageLevel)
    val lda = LDA(this, docs, algo)
    for (i <- 1 to burnIn) {
      lda.gibbsSampling(i, inferenceOnly=true)
    }
    lda.runSum(isDocId, totalIter - burnIn, inferenceOnly=true)
  }

  def toLocalLDAModel: LocalLDAModel = {
    val ttcs = Array.fill(numTerms.toInt)(HashVector.zeros[Count](numTopics))
    termTopicCounters.collect().foreach(t => ttcs(t._1.toInt) :+= t._2)
    new LocalLDAModel(ttcs, numTopics, numTerms, numTokens, alpha, beta, alphaAS)
  }

  override def save(sc: SparkContext, path: String): Unit = save(sc, path, isTransposed=false)

  def save(isTransposed: Boolean): Unit = {
    val sc = termTopicCounters.context
    val outputPath = sc.getConf.get(cs_outputpath)
    save(sc, outputPath, isTransposed)
  }

  /**
   * @param sc           Spark context to get HDFS env from
   * @param path         output path
   * @param isTransposed libsvm when `isTransposed` is false, the format of each line:
   *                     termId  \grave{topicId}:counter \grave{topicId}:counter...,
   *                     in which \grave{topicId} = topicId + 1
   *                     otherwise:
   *                     topicId \grave{termId}:counter \grave{termId}:counter...,
   *                     in which \grave{termId}= termId + 1
   */
  def save(sc: SparkContext, path: String, isTransposed: Boolean): Unit = {
    val metadata = compact(render(("class" -> sv_classNameV1_0) ~ ("version" -> sv_formatVersionV1_0) ~
      ("alpha" -> alpha) ~ ("beta" -> beta) ~ ("alphaAS" -> alphaAS) ~
        ("numTopics" -> numTopics) ~ ("numTerms" -> numTerms) ~ ("numTokens" -> numTokens) ~
        ("isTransposed" -> isTransposed)))
    val rdd = if (isTransposed) {
      termTopicCounters.flatMap { case (termId, vector) =>
        vector.activeIterator.map { case (topicId, cn) =>
          val z = HashVector.zeros[Count](numTerms.toInt)
          z(termId.toInt) = cn
          (topicId.toLong, z)
        }
      }.reduceByKey(_ :+= _)
    } else {
      termTopicCounters
    }

    val saveAsSolid = sc.getConf.getBoolean(cs_saveAsSolid, false)
    if (saveAsSolid) {
      val metadata_line = metadata.replaceAll("\n", "")
      val rdd_txt = rdd.map { case (id, vector) =>
        val list = vector.activeIterator.toList.sortWith((a, b) => a._2 > b._2)
        id.toString + "\t" + list.map(item => item._1 + ":" + item._2).mkString("\t")
      }
      LoaderUtils.RDD2HDFSFile[String](sc, rdd_txt, path, metadata_line, t => t)
    } else {
      sc.parallelize(Seq(metadata), 1).saveAsTextFile(LoaderUtils.metadataPath(path))
      // save model with the topic or word-term descending order
      rdd.map { case (id, vector) =>
        val list = vector.activeIterator.toList.sortWith((a, b) => a._2 > b._2)
        (NullWritable.get(), new Text(id + "\t" + list.map(item => item._1 + ":" + item._2).mkString("\t")))
      }.saveAsHadoopFile[TextOutputFormat[NullWritable, Text]](LoaderUtils.dataPath(path))
    }
  }

  override protected def formatVersion: String = sv_formatVersionV1_0
}

object LDAModel extends Loader[DistributedLDAModel] {
  type MetaT = (Int, Int, Long, Double, Double, Double, Boolean)

  override def load(sc: SparkContext, path: String): DistributedLDAModel = {
    val (loadedClassName, version, metadata) = LoaderUtils.loadMetadata(sc, path)
    val dataPath = LoaderUtils.dataPath(path)
    if (loadedClassName == sv_classNameV1_0 && version == sv_formatVersionV1_0) {
      val metas = parseMeta(metadata)
      val rdd = sc.textFile(dataPath).map(line => parseLine(metas, line))
      loadLDAModel(metas, rdd)
    } else {
      throw new Exception(s"LDAModel.load did not recognize model with (className, format version):" +
        s"($loadedClassName, $version). Supported: ($sv_classNameV1_0, $sv_formatVersionV1_0)")
    }
  }

  def loadFromSolid(sc: SparkContext, path: String): DistributedLDAModel = {
    val (metas, rdd) = LoaderUtils.HDFSFile2RDD(sc, path, header => parseMeta(parse(header)), parseLine)
    loadLDAModel(metas, rdd)
  }

  def parseMeta(metadata: JValue): MetaT = {
    implicit val formats = DefaultFormats
    val alpha = (metadata \ "alpha").extract[Double]
    val beta = (metadata \ "beta").extract[Double]
    val alphaAS = (metadata \ "alphaAS").extract[Double]
    val numTopics = (metadata \ "numTopics").extract[Int]
    val numTerms = (metadata \ "numTerms").extract[Int]
    val numTokens = (metadata \ "numTokens").extract[Int]
    val isTransposed = (metadata \ "isTransposed").extract[Boolean]
    (numTopics, numTerms, numTokens, alpha, beta, alphaAS, isTransposed)
  }

  def parseLine(metas: MetaT, line: String): (Long, HashVector[Count]) = {
    val numTopics = metas._1
    val numTerms = metas._2
    val isTransposed = metas._7
    val numSize = if (isTransposed) numTerms else numTopics
    val sv = HashVector.zeros[Count](numSize)
    val arr = line.split("\t")
    arr.tail.foreach { sub =>
      val Array(index, value) = sub.split(":")
      sv(index.toInt) = value.toInt
    }
    // sv.compact()
    (arr.head.toLong, sv)
  }

  def loadLDAModel(metas: MetaT, rdd: RDD[(Long, HashVector[Count])]): DistributedLDAModel = {
    val (numTopics, numTerms, numTokens, alpha, beta, alphaAS, isTransposed) = metas
    val termCnts = if (isTransposed) {
      rdd.flatMap {
        case (topicId, vector) => vector.activeIterator.map {
          case (termId, cn) =>
            val z = HashVector.zeros[Count](numTopics)
            z(topicId.toInt) = cn
            (termId.toLong, z)
        }
      }.reduceByKey(_ :+= _)
      //  .map {
      //  t => t._2.compact(); t
      // }
    } else {
      rdd
    }
    val storageLevel = StorageLevel.MEMORY_AND_DISK_SER
    termCnts.persist(storageLevel)
    new DistributedLDAModel(termCnts, numTopics, numTerms, numTokens, alpha, beta, alphaAS, storageLevel)
  }

  def loadLocalLDAModel(filePath: String): LocalLDAModel = {
    val file: File = new File(filePath)
    require(file.exists, s"model file $filePath does not exist")
    require(file.isFile, s"model file $filePath is not a normal file")
    val lines = Files.readLines(file, Charsets.UTF_8)
    val Array(sNumTopics, sNumTerms, sNumTokens, sAlpha, sBeta, sAlphaAS) = lines.get(0).split(" ")
    val numTopics = sNumTopics.toInt
    val numTerms = sNumTerms.toInt
    val termCnts = Array.fill(numTerms)(HashVector.zeros[Count](numTopics))
    val iter = lines.listIterator(1)
    while (iter.hasNext) {
      val line = iter.next.trim
      if (!line.isEmpty && !line.startsWith("#")) {
        val its = line.split(" ")
        val offset = its.head.toInt
        val sv = termCnts(offset)
        its.tail.foreach { s =>
          val Array(index, value) = s.split(":")
          sv(index.toInt) = value.toInt
        }
        // sv.compact()
      }
    }
    new LocalLDAModel(termCnts, numTopics, numTerms, sNumTokens.toLong,
      sAlpha.toDouble, sBeta.toDouble, sAlphaAS.toDouble)
  }
}
