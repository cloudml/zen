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

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, sum}
import com.github.cloudml.zen.ml.clustering.LDADefines._
import com.github.cloudml.zen.ml.clustering.algorithm.ZenLDA
import com.github.cloudml.zen.ml.sampler._
import com.github.cloudml.zen.ml.util._
import com.google.common.base.Charsets
import com.google.common.io.Files
import org.apache.hadoop.fs.{FileUtil, Path}
import org.apache.spark.Partitioner._
import org.apache.spark.SparkContext
import org.apache.spark.graphx2._
import org.apache.spark.mllib.util.{Loader, Saveable}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.collection.AppendOnlyMap
import org.json4s.JsonDSL._
import org.json4s._
import org.json4s.jackson.JsonMethods._


class LocalLDAModel(@transient val termTopicsArr: Array[Nwk],
  val numTopics: Int,
  val numTerms: Int,
  val numTokens: Long,
  val alpha: Double,
  val beta: Double,
  val alphaAS: Double) extends Serializable {

  @transient val topicCounters = collectTopicCounters()
  @transient val numThreads = Runtime.getRuntime.availableProcessors
  @transient val algo = new ZenLDA(numTopics, numThreads)

  @transient val alphaSum = alpha * numTopics
  @transient val betaSum = beta * numTerms
  @transient val alphaRatio = algo.calc_alphaRatio(alphaSum, numTokens, alphaAS)
  @transient val denoms = algo.calc_denoms(topicCounters, betaSum)
  @transient val alphak_denoms = algo.calc_alphak_denoms(denoms, alphaAS, betaSum, alphaRatio)
  @transient val beta_denoms = algo.calc_beta_denoms(denoms, beta)

  @transient lazy val termDistCache = new AppendOnlyMap[Int,
    SoftReference[AliasTable[Double]]](numTerms / 2)
  @transient lazy val global = {
    val table = new AliasTable[Double]
    algo.resetDist_abDense(table, alphak_denoms, beta)
  }

  private def collectTopicCounters(): BDV[Count] = {
    termTopicsArr.foldLeft(BDV.zeros[Count](numTopics))(_ :+= _)
  }

  /**
   * inference interface
   * @param doc the doc to be inferred
   * @param runIter overall iterations
   * @param burnIn burn-in iterations
   */
  def inference(
    doc: BSV[Count],
    burnIn: Int = 5,
    runIter: Int = 5): BSV[Double] = {
    require(runIter > 0, "totalIter is less than 1")
    require(burnIn > 0, "burnInIter is less than 1")
    val gen = new XORShiftRandom
    val topicDist = BSV.zeros[Int](numTopics)
    val tokens = vector2Array(doc)
    val topics = new Array[Int](tokens.length)
    var docTopics = uniformDistSampler(gen, tokens, topics, numTopics)
    val docCdf = new CumulativeDist[Double].reset(numTopics)
    for (i <- 1 to burnIn + runIter) {
      docTopics = sampleDoc(gen, docTopics, tokens, topics, docCdf)
      if (i > burnIn) topicDist :+= docTopics
    }
    val pSum = sum(topicDist).toDouble
    topicDist.mapValues(_ / pSum)
  }

  private[ml] def vector2Array(bow: BSV[Count]): Array[Int] = {
    val docLen = sum(bow)
    val sent = new Array[Int](docLen)
    var offset = 0
    bow.activeIterator.filter(_._2 > 0).foreach { case (term, cnt) =>
      for (i <- 0 until cnt) {
        sent(offset) = term
        offset += 1
      }
    }
    sent
  }

  private[ml] def sampleDoc(gen: Random,
    docTopics: Ndk,
    tokens: Array[Int],
    topics: Array[Int],
    docCdf: CumulativeDist[Double]): Ndk = {
    var i = 0
    while (i < topics.length) {
      val termId = tokens(i)
      val termTopics = termTopicsArr(termId)
      val termDist = wSparseCached(termDistCache, termTopics, alphak_denoms, termId)
      val denseTermTopics = toBDV(termTopics)
      val termBeta_denoms = algo.calc_termBeta_denoms(denoms, beta_denoms, termTopics)
      val topic = topics(i)
      docTopics(topic) -= 1
      if (docTopics(topic) == 0) {
        docTopics.compact()
      }
      algo.resetDist_dwbSparse_wOpt(docCdf, termBeta_denoms, docTopics)
      val newTopic = algo.tokenSampling(gen, global, termDist, docCdf, denseTermTopics, topic)
      topics(i) = newTopic
      docTopics(newTopic) += 1
      i += 1
    }
    docTopics
  }

  private[ml] def wSparseCached(cacheMap: AppendOnlyMap[Int, SoftReference[AliasTable[Double]]],
    termTopics: Nwk,
    alphaK_denoms: BDV[Double],
    termId: Int): AliasTable[Double] = {
    if (termTopics.activeSize == 0) return null
    var w = cacheMap(termId)
    if (w == null || w.get() == null) {
      val table = new AliasTable[Double]
      algo.resetDist_waSparse(table, alphaK_denoms, termTopics)
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
    termTopicsArr.zipWithIndex.foreach {
      case (sv, index) =>
        val list = sv.activeIterator.filter(_._2 != 0).map(t => s"${t._1}:${t._2}").mkString(" ")
        fw.write(s"$index $list\n")
    }
    fw.close()
  }
}

class DistributedLDAModel(@transient val termTopicsRDD: RDD[NwkPair],
  val numTopics: Int,
  val numTerms: Int,
  val numTokens: Long,
  val alpha: Double,
  val beta: Double,
  val alphaAS: Double,
  var storageLevel: StorageLevel) extends Serializable with Saveable {

  @transient lazy val totalTopicCounter = termTopicsRDD.map(_._2)
    .aggregate(BDV.zeros[Count](numTopics))(_ :+= _, _ :+= _)
  @transient val algo = new ZenLDA(numTopics, 1)

  /**
   * inference interface
   * @param bowDocs   tuple pair: (dicId, Vector), in which 'docId' is unique
   *                  recommended storage level: StorageLevel.MEMORY_AND_DISK
   * @param runIter   overall iterations
   * @param burnIn    previous burnIn iters results will discard
   */
  def inference(bowDocs: RDD[BOW],
    runIter: Int = 25,
    burnIn: Int = 22): RDD[(VertexId, BSV[Double])] = {
    require(burnIn > 0, "burnIn is less than 1")
    require(runIter > 0, "runIter is less than 1")
    val docs = LDA.initializeCorpusEdges(bowDocs, "bow", numTopics, algo, storageLevel)
    val lda = LDA(this, docs, algo)
    for (i <- 1 to burnIn) {
      lda.gibbsSampling(i)
    }
    // val topicDist = lda.runSum(isDocId, runIter)
    val topicDist = lda.termVertices.mapPartitions(iter => {
      val decomp = new BVDecompressor(numTopics)
      iter.map(Function.tupled((vid, counter) =>
        (vid, decomp.CV2BV(counter))
      ))
    }, preservesPartitioning = true)
    topicDist.mapValues(v => {
      val dist = v.asInstanceOf[BSV[Count]]
      val pSum = sum(dist).toDouble
      dist.mapValues(_ / pSum)
    })
  }

  def toLocalLDAModel: LocalLDAModel = {
    val ttcs: Array[Nwk] = Array.fill(numTerms.toInt)(BSV.zeros[Count](numTopics))
    termTopicsRDD.collect().foreach(t => ttcs(t._1.toInt) :+= t._2)
    new LocalLDAModel(ttcs, numTopics, numTerms, numTokens, alpha, beta, alphaAS)
  }

  def save(): Unit = {
    val sc = termTopicsRDD.context
    val outputPath = sc.getConf.get(cs_outputpath)
    save(sc, outputPath)
  }

  /**
    * Save model in libsvm format. When `isTransposed` is false, the format of each line:
    *   termId  \grave{topicId}:counter \grave{topicId}:counter...,
    * in which \grave{topicId} = topicId + 1
    * otherwise:
    *   topicId \grave{termId}:counter \grave{termId}:counter...,
    * in which \grave{termId}= termId + 1
    * @param sc           Spark context to get HDFS env from
    * @param path         output path
    */
  override def save(sc: SparkContext, path: String): Unit = {
    val isTransposed = sc.getConf.getBoolean(cs_saveTransposed, true)
    var json = ("class" -> sv_classNameV2_0) ~ ("version" -> sv_formatVersionV2_0) ~
      ("alpha" -> alpha) ~ ("beta" -> beta) ~ ("alphaAS" -> alphaAS) ~
      ("numTopics" -> numTopics) ~ ("numTerms" -> numTerms) ~ ("numTokens" -> numTokens) ~
      ("isTransposed" -> isTransposed)
    val rdd = if (isTransposed) {
      val maxTerms = termTopicsRDD.map(_._1).filter(isTermId).max().toInt + 1
      json = json ~ ("maxTerms" -> maxTerms)
      val partitioner = defaultPartitioner(termTopicsRDD)
      termTopicsRDD.flatMap {
        case (termId, vector) =>
          val term = termId.toInt
          vector.activeIterator.map {
            case (topic, cnt) => (topic.toLong, (term, cnt))
          }
      }.aggregateByKey[BV[Count]](BSV.zeros[Count](maxTerms), partitioner)((agg, t) => {
        agg(t._1) += t._2
        agg
      }, _ :+= _)
    } else {
      termTopicsRDD
    }
    rdd.persist(storageLevel)
    val metadata = compact(render(json))

    val saveAsSolid = sc.getConf.getBoolean(cs_saveAsSolid, false)
    val savPath = if (saveAsSolid) new Path(path + ".sav") else new Path(path)
    val savDir = savPath.toUri.toString
    val metaDir = LoaderUtils.metadataPath(savDir)
    val dataDir = LoaderUtils.dataPath(savDir)
    val fs = SparkUtils.getFileSystem(sc.getConf, savPath)

    fs.delete(savPath, true)
    sc.parallelize(Seq(metadata), 1).saveAsTextFile(metaDir)
    // save model with the topic or word-term descending order
    rdd.map { case (id, vector) =>
      val list = vector.activeIterator.filter(_._2 > 0).toSeq.sortBy(_._2).reverse
        .map(t => s"${t._1}:${t._2}").mkString("\t")
      s"$id\t$list"
    }.saveAsTextFile(dataDir)
    if (saveAsSolid) {
      val cpmgPath = new Path(path + ".cpmg")
      fs.delete(cpmgPath, true)
      var suc = fs.rename(new Path(metaDir + "/part-00000"), new Path(dataDir + "/_meta"))
      if (suc) {
        suc = FileUtil.copyMerge(fs, new Path(dataDir), fs, cpmgPath, false, sc.hadoopConfiguration, null)
      }
      if (suc) {
        suc = fs.rename(cpmgPath, new Path(path))
      }
      fs.delete(savPath, true)
      fs.delete(cpmgPath, true)
      if (!suc) {
        throw new IOException("Save model error!")
      }
    }
  }

  override protected def formatVersion: String = sv_formatVersionV2_0
}

object LDAModel extends Loader[DistributedLDAModel] {
  type MetaT = (Int, Int, Long, Double, Double, Double, Boolean, Option[Int])

  override def load(sc: SparkContext, path: String): DistributedLDAModel = {
    val (loadedClassName, version, metadata) = LoaderUtils.loadMetadata(sc, path)
    val dataPath = LoaderUtils.dataPath(path)
    if (loadedClassName == sv_classNameV2_0 && version == sv_formatVersionV2_0) {
      val metas = parseMeta(metadata)
      var rdd = sc.textFile(dataPath).map(line => parseLine(metas, line))
      rdd = sc.getConf.getOption(cs_numPartitions).map(_.toInt) match {
        case Some(np) => rdd.coalesce(np, shuffle=true)
        case None => rdd
      }
      loadLDAModel(metas, rdd)
    } else {
      throw new Exception(s"LDAModel.load did not recognize model with (className, format version):" +
        s"($loadedClassName, $version). Supported: ($sv_classNameV2_0, $sv_formatVersionV2_0)")
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
    val maxTerms = (metadata \ "maxTerms").extractOpt[Int]
    (numTopics, numTerms, numTokens, alpha, beta, alphaAS, isTransposed, maxTerms)
  }

  def parseLine(metas: MetaT, line: String): BOW = {
    val numTopics = metas._1
    val isTransposed = metas._7
    val maxTerms = metas._8
    val numSize = if (isTransposed) maxTerms.get else numTopics
    val sv = BSV.zeros[Count](numSize)
    val arr = line.split("\t")
    arr.tail.foreach(sub => {
      val Array(index, value) = sub.split(":")
      sv(index.toInt) = value.toInt
    })
    (arr.head.toLong, sv)
  }

  def loadLDAModel(metas: MetaT, rdd: RDD[BOW]): DistributedLDAModel = {
    val (numTopics, numTerms, numTokens, alpha, beta, alphaAS, isTransposed, _) = metas
    val termCnts = if (isTransposed) {
      val partitioner = defaultPartitioner(rdd)
      rdd.flatMap {
        case (topicId, vector) =>
          val topic = topicId.toInt
          vector.activeIterator.map {
            case (term, cnt) => (term.toLong, (topic, cnt))
          }
      }.aggregateByKey(BSV.zeros[Count](numTopics), partitioner)((agg, t) => {
        agg(t._1) += t._2
        agg
      }, _ :+= _)
    } else {
      rdd
    }
    val storageLevel = StorageLevel.MEMORY_AND_DISK
    termCnts.persist(storageLevel)
    new DistributedLDAModel(termCnts.asInstanceOf[RDD[(VertexId, Nwk)]], numTopics, numTerms, numTokens,
      alpha, beta, alphaAS, storageLevel)
  }

  def loadLocalLDAModel(filePath: String): LocalLDAModel = {
    val file: File = new File(filePath)
    require(file.exists, s"model file $filePath does not exist")
    require(file.isFile, s"model file $filePath is not a normal file")
    val lines = Files.readLines(file, Charsets.UTF_8)
    val Array(sNumTopics, sNumTerms, sNumTokens, sAlpha, sBeta, sAlphaAS) = lines.get(0).split(" ")
    val numTopics = sNumTopics.toInt
    val numTerms = sNumTerms.toInt
    val termCnts: Array[Nwk] = Array.fill(numTerms)(BSV.zeros[Count](numTopics))
    val iter = lines.listIterator(1)
    while (iter.hasNext) {
      val line = iter.next.trim
      if (!line.isEmpty && !line.startsWith("#")) {
        val its = line.split(" ")
        val offset = its.head.toInt
        val sv = termCnts(offset)
        its.tail.foreach(s => {
          val Array(index, value) = s.split(":")
          sv(index.toInt) = value.toInt
        })
      }
    }
    new LocalLDAModel(termCnts, numTopics, numTerms, sNumTokens.toLong,
      sAlpha.toDouble, sBeta.toDouble, sAlphaAS.toDouble)
  }
}
