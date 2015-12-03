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

package com.github.cloudml.zen.ml.sampler

import java.util.Random
import scala.reflect.ClassTag

import CumulativeDist._

import breeze.linalg.StorageVector
import spire.math.{Numeric => spNum}


class CumulativeDist[@specialized(Double, Int, Float, Long) T: ClassTag](implicit ev: spNum[T])
  extends DiscreteSampler[T] with Serializable {
  var _cdf: Array[T] = _
  var _space: Array[Int] = _
  var _used: Int = _

  def length: Int = _cdf.length

  def size: Int  = length

  def used: Int = _used

  def norm: T = {
    if (_used == 0) {
      ev.zero
    } else {
      _cdf(_used - 1)
    }
  }

  def sampleRandom(gen: Random)(implicit gev: spNum[T]): Int = {
    val u = gen.nextDouble() * gev.toDouble(_cdf(_used - 1))
    sampleFrom(gev.fromDouble(u), gen)
  }

  def sampleFrom(base: T, gen: Random): Int = {
    // assert(ev.lt(base, _cdf(_used - 1)))
    if (_used == 1) {
      _space(0)
    } else {
      val i = binarySelect(_cdf, base, 0, _used, greater=true)
      _space(i)
    }
  }

  def update(state: Int, value: => T): Unit = {}

  def deltaUpdate(state: Int, delta: => T): Unit = {}

  def resetDist(probs: Array[T], space: Array[Int], psize: Int): CumulativeDist[T] = {
    resetDist(space.iterator.zip(probs.iterator), psize)
  }

  def resetDist(distIter: Iterator[(Int, T)], psize: Int): CumulativeDist[T] = {
    reset(psize)
    var sum = ev.zero
    var i = 0
    while (i < psize) {
      val (state, prob) = distIter.next()
      sum = ev.plus(sum, prob)
      _cdf(i) = sum
      _space(i) = state
      i += 1
    }
    this
  }

  def reset(newSize: Int): CumulativeDist[T] = {
    if (_cdf == null || _cdf.length < newSize) {
      _cdf = new Array[T](newSize)
      _space = new Array[Int](newSize)
    }
    _used = newSize
    this
  }

  def data: Array[T] = _cdf
}

object CumulativeDist {
  def generateCdf[@specialized(Double, Int, Float, Long) T: ClassTag: spNum]
  (sv: StorageVector[T]): CumulativeDist[T] = {
    val used = sv.activeSize
    val cdf = new CumulativeDist[T]
    cdf.resetDist(sv.activeIterator, used)
  }

  def binarySelect[@specialized(Double, Int, Float, Long) T](arr: Array[T], key: T,
    begin: Int, end: Int, greater: Boolean)(implicit ev: spNum[T]): Int = {
    if (begin == end) {
      return if (greater) end else begin - 1
    }
    var b = begin
    var e = end - 1

    var mid: Int = (e + b) >> 1
    while (b <= e) {
      mid = (e + b) >> 1
      val v = arr(mid)
      if (ev.lt(v, key)) {
        b = mid + 1
      } else if (ev.gt(v, key)) {
        e = mid - 1
      } else {
        return mid
      }
    }
    val v = arr(mid)
    mid = if ((greater && ev.gteqv(v, key)) || (!greater && ev.lteqv(v, key))) {
      mid
    } else if (greater) {
      mid + 1
    } else {
      mid - 1
    }

//    if (greater) {
//      if (mid < end) assert(ev.gteqv(arr(mid), key))
//      if (mid > 0) assert(ev.lteqv(arr(mid - 1), key))
//    } else {
//      if (mid > 0) assert(ev.lteqv(arr(mid), key))
//      if (mid < end - 1) assert(ev.gteqv(arr(mid + 1), key))
//    }
    mid
  }
}
