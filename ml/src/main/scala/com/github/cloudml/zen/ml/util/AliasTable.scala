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

import breeze.linalg.{Vector => BV}


class AliasTable[@specialized(Double, Int, Float, Long) T: ClassTag](initUsed: Int)
  (implicit num: Numeric[T]) extends DiscreteSampler[T] with Serializable {
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
    @inline def isClose(a: Double, b: Double): Boolean = math.abs(a - b) <= (1e-8 + math.abs(a) * 1e-6)
    var sum = num.zero
    var bin = 0
    var i = 0
    while (i < probs.length) {
      val prob = probs(i)
      if (num.gt(prob, num.zero)) {
        sum = num.plus(sum, prob)
        bin += 1
      }
      i += 1
    }
    sum = probs.sum
    reset(bin)
    var ls = 0
    var le = 0
    var he = bin
    val sq = new Array[Int](bin)
    val pq = new Array[T](bin)
    var qi = 0
    val scale = num.fromInt(bin)
    i = 0
    while (i < bin) {
      val prob = probs(i)
      if (num.gt(prob, num.zero)) {
        val pscale = num.times(prob, scale)
        val state = if (space == null) i else space(i)
        if (num.lt(pscale, sum)) {
          _l(le) = state
          _p(le) = pscale
          le += 1
        } else {
          sq(qi) = state
          pq(qi) = pscale
          qi += 1
        }
      }
      i += 1
    }
    i = 0
    while (i < qi) {
      val state = sq(i)
      val prob = pq(i)
      if (num.lt(prob, sum)) {
        _l(le) = state
        _p(le) = prob
        le += 1
        i += 1
      } else if (ls < le) {
        _h(ls) = state
        val split = _p(ls)
        ls += 1
        val prest = num.minus(prob, num.minus(sum, split))
        pq(i) = prest
      } else {
        assert(isClose(num.toDouble(prob), num.toDouble(sum)), s"prob=$prob, sum=$sum")
        he -= 1
        _l(he) = state
        _h(he) = state
        i += 1
      }
    }
    assert(le - ls <= 1 && math.abs(he - le) <= 1 && math.abs(he - ls) <= 1, s"le=$le, ls=$ls, he=$he")
    setNorm(sum)
  }

  type Pair = (Int, T)

  def resetDist(distIter: Iterator[(Int, T)], used: Int): this.type = synchronized {    val dist = distIter.toList
    val sum = dist.map(_._2).sum
    reset(used)
    val (loList, hiList) = dist.map(t => (t._1, num.times(t._2, num.fromInt(used))))
      .partition(t => num.lt(t._2, sum))
    var ls = 0
    var le = 0
    var end = used
    @inline def isClose(a: Double, b: Double): Boolean = math.abs(a - b) <= (1e-8 + math.abs(a) * 1e-6)
    def putAlias(list: List[Pair], rest: List[Pair]): List[Pair] = list match {
      case Nil => rest
      case (t, pt) :: rlist if num.lt(pt, sum) =>
        _l(le) = t
        _p(le) = pt
        le += 1
        putAlias(rlist, rest)
      case (t, pt) :: rlist if ls < le =>
        _h(ls) = t
        val pl = _p(ls)
        ls += 1
        val pd = num.minus(pt, num.minus(sum, pl))
        putAlias(List((t, pd)) ++ rlist, rest)
      case (t, pt) :: rlist=>
        putAlias(rlist, rest ++ List((t, pt)))
    }
    def putRest(rest: List[Pair]): Unit = rest match {
      case Nil => Unit
      case (t, pt) :: rrest =>
        assert(isClose(num.toDouble(pt), num.toDouble(sum)))
        end -= 1
        _l(end) = t
        _h(end) = t
        putRest(rrest)
    }
    putRest(putAlias(hiList, putAlias(loList, List())))
    assert(le - ls <= 1 && math.abs(end - le) <= 1 && math.abs(end - ls) <= 1)
    setNorm(sum)
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
