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
import scala.collection.mutable.ArrayBuffer
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

  def resetDist(probs: Array[T], space: Array[Int]): this.type = synchronized {
    val used = probs.length
    reset(used)
    val norm = probs.sum
    var loStart = 0
    var loEnd = 0
    var hiEnd = used
    val hq = new Array[Int](used)
    var hi = 0
    val scale = num.fromInt(used)
    var i = 0
    while (i < used) {
      val prob = num.times(probs(i), scale)
      if (num.lt(prob, norm)) {
        val state = space(i)
        _l(loEnd) = state
        _p(loEnd) = prob
        loEnd +=1
      } else {
        probs(i) = prob
        hq(hi) = i
        hi += 1
      }
      i += 1
    }
    var j = 0
    while (j < hi) {
      val i = hq(j)
      val state = space(i)
      val prob = probs(i)
      if (num.lt(prob, norm)) {
        _l(loEnd) = state
        _p(loEnd) = scale
        loEnd += 1
        j += 1
      } else if (loStart < loEnd) {
        _h(loStart) = state
        val split = _p(loStart)
        val pleft = num.minus(scale, num.minus(norm, split))
        probs(i) = pleft
        loStart += 1
      } else {
        hiEnd -= 1
        _l(hiEnd) = state
        _h(hiEnd) = state
        j += 1
      }
    }
    assert(loEnd - loStart <= 1 && math.abs(hiEnd - loEnd) <= 1 && math.abs(hiEnd - loStart) <= 1)
    setNorm(norm)
  }

  def resetDist(distIter: Iterator[(Int, T)], used: Int): this.type = synchronized {
    val (probsIter, spaceIter) = distIter.duplicate
    val probs = probsIter.map(_._2).toArray
    val space = spaceIter.map(_._1).toArray
    resetDist(probs, space)
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
