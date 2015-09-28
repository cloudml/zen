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
import scala.collection.mutable
import scala.reflect.ClassTag

import breeze.linalg.{Vector => BV}


class AliasTable[@specialized(Double, Int, Float, Long) T: ClassTag](initUsed: Int)
  (implicit num: Numeric[T])
  extends DiscreteSampler[T] with Serializable {
  type Pair = (Int, T)

  private var _l: Array[Int] = new Array[Int](initUsed)
  private var _h: Array[Int] = new Array[Int](initUsed)
  private var _p: Array[T] = new Array[T](initUsed)
  private var _used = initUsed
  private var _norm = num.zero

  def l: Array[Int] = _l

  def h: Array[Int] = _h

  def p: Array[T] = _p

  def used: Int = _used

  def length: Int = _l.length

  def size: Int = length

  def norm: T = _norm

  def sampleRandom(gen: Random): Int = {
    if (_used == 1) {
      _l(0)
    } else {
      val bin = gen.nextInt(_used)
      val prob = _p(bin)
      if (gen.nextDouble() * num.toDouble(_norm) < num.toDouble(prob)) {
        _l(bin)
      } else {
        _h(bin)
      }
    }
  }

  def sampleFrom(base: T, gen: Random): Int = {
    assert(num.lt(base, _norm))
    if (_used == 1) {
      _l(0)
    } else {
      val bin = gen.nextInt(_used)
      val prob = _p(bin)
      if (num.lt(base, prob)) {
        _l(bin)
      } else {
        _h(bin)
      }
    }
  }

  def update(state: Int, value: T): Unit = {}

  def deltaUpdate(state: Int, delta: T): Unit = {}

  def resetDist(distIter: Iterator[(Int, T)], used: Int): this.type = synchronized {
    val dist = distIter.filter(Function.tupled((_, prob) => num.gt(prob, num.zero))).toArray
    reset(dist.length)
    val norm = dist.iterator.map(_._2).sum
    var loStart = 0
    var loEnd = 0
    var hiEnd = _used
    val hs = new mutable.ArrayStack[Pair]
    dist.foreach(Function.tupled((state, prob) => {
      val scale = num.times(prob, num.fromInt(_used))
      if (num.lt(scale, norm)) {
        _l(loEnd) = state
        _p(loEnd) = scale
        loEnd += 1
      } else {
        hs.push((state, scale))
      }
    }))
    while (hs.nonEmpty) {
      val (state, scale) = hs.pop()
      if (num.lt(scale, norm)) {
        _l(loEnd) = state
        _p(loEnd) = scale
        loEnd += 1
      } else if (loStart < loEnd) {
        _h(loStart) = state
        val split = _p(loStart)
        val scale_left = num.minus(scale, num.minus(norm, split))
        hs.push((state, scale_left))
        loStart += 1
      } else {
        hiEnd -= 1
        _l(hiEnd) = state
        _h(hiEnd) = state
      }
    }
    // assert(loEnd - loStart <= 1 && math.abs(hiEnd - loEnd) <= 1 && math.abs(hiEnd - loStart) <= 1)
    setNorm(norm)
  }

  private def reset(newSize: Int): this.type = {
    if (_l.length < newSize) {
      _l = new Array[Int](newSize)
      _h = new Array[Int](newSize)
      _p = new Array[T](newSize)
    }
    _used = newSize
    _norm = num.zero
    this
  }

  private def setNorm(norm: T): this.type = {
    _norm = norm
    this
  }
}

object AliasTable {
  def generateAlias[@specialized(Double, Int, Float, Long) T: ClassTag: Numeric](sv: BV[T]): AliasTable[T] = {
    val used = sv.activeSize
    val table = new AliasTable[T](used)
    table.resetDist(sv.activeIterator, used)
  }
}
