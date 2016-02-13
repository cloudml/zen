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

package com.github.cloudml.zen.ml.util

import java.util

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}
import me.lemire.integercompression._
import me.lemire.integercompression.differential._


class CompressedVector(val used: Int,
  val cdata: Array[Int],
  val cindex: Array[Int]) extends Serializable

// optimized for performance, not thread-safe
class BVCompressor(numTopics: Int) {
  val dataCodec = new SkippableComposition(new BinaryPacking, new VariableByte)
  val indexCodec = new SkippableIntegratedComposition(new IntegratedBinaryPacking, new IntegratedVariableByte)
  val buf = new Array[Int](numTopics + 1024)
  val inPos = new IntWrapper
  val outPos = new IntWrapper
  val initValue = new IntWrapper

  def BV2CV(bv: BV[Int]): CompressedVector = bv match {
    case v: BDV[Int] =>
      val cdata = compressData(v.data, numTopics)
      new CompressedVector(numTopics, cdata, null)
    case v: BSV[Int] =>
      val used = v.used
      val index = v.index
      val data = v.data
      if (used <= 4) {
        new CompressedVector(used, data, index)
      } else {
        val cdata = compressData(data, used)
        val cindex = compressIndex(index, used)
        new CompressedVector(used, cdata, cindex)
      }
  }

  def compressData(data: Array[Int], len: Int): Array[Int] = {
    inPos.set(0)
    outPos.set(0)
    dataCodec.headlessCompress(data, inPos, len, buf, outPos)
    util.Arrays.copyOf(buf, outPos.get)
  }

  def compressIndex(index: Array[Int], len: Int): Array[Int] = {
    buf(0) = index.length
    inPos.set(0)
    outPos.set(1)
    initValue.set(0)
    indexCodec.headlessCompress(index, inPos, len, buf, outPos, initValue)
    util.Arrays.copyOf(buf, outPos.get)
  }
}

// optimized for performance, not thread-safe
class BVDecompressor(numTopics: Int) {
  val dataCodec = new SkippableComposition(new BinaryPacking, new VariableByte)
  val indexCodec = new SkippableIntegratedComposition(new IntegratedBinaryPacking, new IntegratedVariableByte)
  val inPos = new IntWrapper
  val outPos = new IntWrapper
  val initValue = new IntWrapper

  def CV2BV(cv: CompressedVector): BV[Int] = {
    val cdata = cv.cdata
    val cindex = cv.cindex
    if (cindex == null) {
      val data = decompressData(cdata, numTopics)
      new BDV(data)
    } else {
      val used = cv.used
      if (used <= 4) {
        new BSV(cindex, cdata, used, numTopics)
      } else {
        val data = decompressData(cdata, used)
        val index= decompressIndex(cindex, used)
        new BSV(index, data, used, numTopics)
      }
    }
  }

  def decompressData(cdata: Array[Int], rawLen: Int): Array[Int] = {
    val data = new Array[Int](rawLen)
    inPos.set(0)
    outPos.set(0)
    dataCodec.headlessUncompress(cdata, inPos, cdata.length, data, outPos, rawLen)
    data
  }

  def decompressIndex(cindex: Array[Int], rawLen: Int): Array[Int] = {
    val index = new Array[Int](rawLen)
    inPos.set(1)
    outPos.set(0)
    initValue.set(0)
    indexCodec.headlessUncompress(cindex, inPos, cindex.length - 1, index, outPos, rawLen, initValue)
    index
  }
}
