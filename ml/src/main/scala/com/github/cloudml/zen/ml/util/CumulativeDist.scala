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

import java.util.Random
import scala.reflect.ClassTag
import CumulativeDist._
import breeze.linalg.{Vector => BV}


private[zen] class CumulativeDist[@specialized(Double, Int, Float, Long) T: ClassTag](dataSize: Int)
  (implicit num: Numeric[T]) extends DiscreteSampler[T] with Serializable {

  private var _cdf = new Array[T](dataSize)
  private var _space = new Array[Int](dataSize)
  private var _used: Int = dataSize

  def length: Int = _cdf.length

  def size: Int  = length

  def used: Int = _used

  def norm: T = {
    if (_used == 0) {
      num.zero
    } else {
      _cdf(_used - 1)
    }
  }

  def sampleRandom(gen: Random): Int = {
    if (_used == 1) {
      _space(0)
    } else {
      val u = gen.nextDouble() * num.toDouble(_cdf(_used - 1))
      val i = binarySelect(_cdf.map(num.toDouble), u, 0, _used, greater=true)
      _space(i)
    }
  }

  def sampleFrom(base: T, gen: Random): Int = {
    assert(num.lt(base, _cdf(_used - 1)))
    if (_used == 1) {
      _space(0)
    } else {
      val i = binarySelect(_cdf, base, 0, _used, greater=true)
      _space(i)
    }
  }

  def update(state: Int, value: T): Unit = {}

  def deltaUpdate(state: Int, delta: T): Unit = {}

  def resetDist(distIter: Iterator[(Int, T)], used: Int): this.type = synchronized {
    reset(used)
    var sum = num.zero
    for (((state, prob), i) <- distIter.zipWithIndex) {
      sum = num.plus(sum, prob)
      _cdf(i) = sum
      _space(i) = state
    }
    this
  }

  // don't use this method unless you know what you are doing
  def directReset(vf: Int => T, used: Int, space: Array[Int]): this.type = synchronized {
    _used = used
    var sum = num.zero
    for (i <- 0 until used) {
      sum = num.plus(sum, vf(i))
      _cdf(i) = sum
    }
    _space = space
    this
  }

  private def reset(newSize: Int): this.type = {
    if (_cdf.length < newSize) {
      _cdf = new Array[T](newSize)
      _space = new Array[Int](newSize)
    }
    _used = newSize
    this
  }

  def data: Array[T] = _cdf
}

private[zen] object CumulativeDist {
  def generateCdf[@specialized(Double, Int, Float, Long) T: ClassTag: Numeric](sv: BV[T]): CumulativeDist[T] = {
    val used = sv.activeSize
    val cdf = new CumulativeDist[T](used)
    cdf.resetDist(sv.activeIterator, used)
  }

  def binarySelect[T](arr: Array[T], key: T, begin: Int, end: Int, greater: Boolean)
    (implicit num: Numeric[T]): Int = {
    if (begin == end) {
      return if (greater) end else begin - 1
    }
    var b = begin
    var e = end - 1

    var mid: Int = (e + b) >> 1
    while (b <= e) {
      mid = (e + b) >> 1
      val v = arr(mid)
      if (num.lt(v, key)) {
        b = mid + 1
      }
      else if (num.gt(v, key)) {
        e = mid - 1
      }
      else {
        return mid
      }
    }
    val v = arr(mid)
    mid = if ((greater && num.gteq(v, key)) || (!greater && num.lteq(v, key))) {
      mid
    }
    else if (greater) {
      mid + 1
    }
    else {
      mid - 1
    }

    if (greater) {
      if (mid < end) assert(num.gteq(arr(mid), key))
      if (mid > 0) assert(num.lteq(arr(mid - 1), key))
    } else {
      if (mid > 0) assert(num.lteq(arr(mid), key))
      if (mid < end - 1) assert(num.gteq(arr(mid + 1), key))
    }
    mid
  }
}
