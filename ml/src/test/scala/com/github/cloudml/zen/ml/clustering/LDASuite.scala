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

import java.io.File
import java.util.Random

import breeze.linalg.functions.euclideanDistance
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV}
import breeze.stats.distributions.Poisson
import com.github.cloudml.zen.ml.util.SharedSparkContext
import com.google.common.io.Files
import org.apache.spark.storage.StorageLevel
import org.scalatest.FunSuite

class LDASuite extends FunSuite with SharedSparkContext {

  import LDASuite._

  test("FastLDA || Gibbs sampling") {
    val model = generateRandomLDAModel(numTopics, numTerms)
    val corpus = sampleCorpus(model, numDocs, numTerms, numTopics)

    val data = sc.parallelize(corpus, 2)
    data.cache()
    val pps = new Array[Double](incrementalLearning)
    val lda = FastLDA(data, numTopics, alpha, beta, alphaAS, storageLevel, partStrategy)
    var i = 0
    val startedAt = System.currentTimeMillis()
    while (i < incrementalLearning) {
      lda.runGibbsSampling(totalIterations, chkptInterval)
      pps(i) = lda.perplexity()
      i += 1
    }

    println((System.currentTimeMillis() - startedAt) / 1e3)
    pps.foreach(println)

    val ppsDiff = pps.init.zip(pps.tail).map { case (lhs, rhs) => lhs - rhs }
    assert(ppsDiff.count(_ > 0).toDouble / ppsDiff.length > 0.6)
    assert(pps.head - pps.last > 0)

    val ldaModel = lda.saveModel()
    val tempDir = Files.createTempDir()
    tempDir.deleteOnExit()
    val path = tempDir.toURI.toString + File.separator + "lda"
    ldaModel.save(sc, path, isTransposed = true, saveSolid = true)
    val sameModel = LDAModel.load(sc, path)
    assert(sameModel.toLocalLDAModel().ttc === ldaModel.toLocalLDAModel().ttc)
    assert(sameModel.alpha === ldaModel.alpha)
    assert(sameModel.beta === ldaModel.beta)
    assert(sameModel.alphaAS === ldaModel.alphaAS)

    val localLdaModel = sameModel.toLocalLDAModel()
    val tempDir2 = Files.createTempDir()
    tempDir2.deleteOnExit()
    val path2 = tempDir2.toString + File.separator + "lda.txt"
    localLdaModel.save(path2)
    val loadLdaModel = LDAModel.loadLocalLDAModel(path2)

    assert(localLdaModel.ttc === loadLdaModel.ttc)
    assert(localLdaModel.alpha === loadLdaModel.alpha)
    assert(localLdaModel.beta === loadLdaModel.beta)
    assert(localLdaModel.alphaAS === loadLdaModel.alphaAS)

  }

  test("LightLDA || Metropolis Hasting sampling") {
    val model = generateRandomLDAModel(numTopics, numTerms)
    val corpus = sampleCorpus(model, numDocs, numTerms, numTopics)

    val data = sc.parallelize(corpus, 2)
    data.cache()
    val pps = new Array[Double](incrementalLearning)
    val lda = LightLDA(data, numTopics, alpha, beta, alphaAS, storageLevel, partStrategy)
    var i = 0
    val startedAt = System.currentTimeMillis()
    while (i < incrementalLearning) {
      lda.runGibbsSampling(totalIterations, chkptInterval)
      pps(i) = lda.perplexity()
      i += 1
    }

    println((System.currentTimeMillis() - startedAt) / 1e3)
    pps.foreach(println)

    val ppsDiff = pps.init.zip(pps.tail).map { case (lhs, rhs) => lhs - rhs }
    assert(ppsDiff.count(_ > 0).toDouble / ppsDiff.length > 0.6)
    assert(pps.head - pps.last > 0)

    val ldaModel = lda.saveModel(3).toLocalLDAModel()
    data.collect().foreach { case (_, sv) =>
      val a = ldaModel.inference(sv)
      val b = ldaModel.inference(sv)
      val sim: Double = euclideanDistance(a, b)
      assert(sim < 0.1)
    }
  }
}

object LDASuite {
  val numTopics = 5
  val numTerms = 1000
  val numDocs = 100
  val expectedDocLength = 300
  val alpha = 0.01f
  val alphaAS = 1f
  val beta = 0.01f
  val totalIterations = 2
  val burnInIterations = 1
  val incrementalLearning = 10
  val partStrategy = "dbh"
  val chkptInterval = 10
  val storageLevel = StorageLevel.MEMORY_AND_DISK

  /**
   * Generate a random LDA model, i.e. the topic-term matrix.
   */
  def generateRandomLDAModel(numTopics: Int, numTerms: Int): Array[BDV[Float]] = {
    val model = new Array[BDV[Float]](numTopics)
    val width = numTerms.toFloat / numTopics
    var topic = 0
    var i = 0
    while (topic < numTopics) {
      val topicCentroid = width * (topic + 1)
      model(topic) = BDV.zeros[Float](numTerms)
      i = 0
      while (i < numTerms) {
        // treat the term list as a circle, so the distance between the first one and the last one
        // is 1, not n-1.
        val distance = Math.abs(topicCentroid - i) % (numTerms / 2)
        // Possibility is decay along with distance
        model(topic)(i) = 1F / (1F + Math.abs(distance))
        i += 1
      }
      topic += 1
    }
    model
  }

  /**
   * Sample one document given the topic-term matrix.
   */
  def ldaSampler(
    model: Array[BDV[Float]],
    topicDist: BDV[Float],
    numTermsPerDoc: Int): Array[Int] = {
    val samples = new Array[Int](numTermsPerDoc)
    val rand = new Random()
    (0 until numTermsPerDoc).foreach { i =>
      samples(i) = multinomialDistSampler(
        rand,
        model(multinomialDistSampler(rand, topicDist))
      )
    }
    samples
  }

  /**
   * Sample corpus (many documents) from a given topic-term matrix.
   */
  def sampleCorpus(
    model: Array[BDV[Float]],
    numDocs: Int,
    numTerms: Int,
    numTopics: Int): Array[(Long, BSV[Int])] = {
    (0 until numDocs).map { i =>
      val rand = new Random()
      val numTermsPerDoc = Poisson.distribution(expectedDocLength).sample()
      val numTopicsPerDoc = rand.nextInt(numTopics / 2) + 1
      val topicDist = BDV.zeros[Float](numTopics)
      (0 until numTopicsPerDoc).foreach { _ =>
        topicDist(rand.nextInt(numTopics)) += 1
      }
      val sv = BSV.zeros[Int](numTerms)
      ldaSampler(model, topicDist, numTermsPerDoc).foreach { term => sv(term) += 1 }
      (i.toLong, sv)
    }.toArray
  }

  /**
   * A multinomial distribution sampler, using roulette method to sample an Int back.
   */
  def multinomialDistSampler(rand: Random, dist: BDV[Float]): Int = {
    val distSum = rand.nextFloat() * breeze.linalg.sum[BDV[Float], Float](dist)

    def loop(index: Int, accum: Float): Int = {
      if (index == dist.length) return dist.length - 1
      val sum = accum - dist(index)
      if (sum <= 0) index else loop(index + 1, sum)
    }

    loop(0, distSum)
  }
}
