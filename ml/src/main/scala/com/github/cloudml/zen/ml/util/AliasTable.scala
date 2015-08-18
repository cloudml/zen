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
import breeze.linalg.{Vector=>BV}


private[zen] class AliasTable[@specialized(Double, Int, Float, Long) T: ClassTag](initUsed: Int)
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
    val bin = gen.nextInt(_used)
    val prob = _p(bin)
    if (gen.nextDouble() * num.toDouble(_norm) < num.toDouble(prob)) {
      _l(bin)
    } else {
      _h(bin)
    }
  }

  def sampleFrom(base: T, gen: Random): Int = {
    assert(num.lt(base, _norm))
    val bin = gen.nextInt(_used)
    val prob = _p(bin)
    if (num.lt(base, prob)) {
      _l(bin)
    } else {
      _h(bin)
    }
  }

  def update(state: Int, delta: T): Unit = {}

  def resetDist(dist: BV[T], sum: T): this.type = {
    val used = dist.activeSize
    reset(used)
    val (loList, hiList) = dist.activeIterator
      .map(t => (t._1, num.times(t._2, num.fromInt(used)))).toList
      .partition(t => num.lt(t._2, sum))
    var ls = 0
    var le = 0
    var end = used
    // @inline def isClose(a: Double, b: Double): Boolean = abs(a - b) <= (1e-8 + abs(a) * 1e-6)
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
        // assert(isClose(pt, sum))
        end -= 1
        _l(end) = t
        _h(end) = t
        putRest(rrest)
    }
    putRest(putAlias(hiList, putAlias(loList, List())))
    // assert(abs(le - ls) <= 1 && abs(end - le) <= 1 && abs(end - ls) <= 1)
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

private[zen] object AliasTable {
  def generateAlias[@specialized(Double, Int, Float, Long) T: ClassTag: Numeric](sv: BV[T]): AliasTable[T] = {
    generateAlias(sv, sv.valuesIterator.sum)
  }

  def generateAlias[@specialized(Double, Int, Float, Long) T: ClassTag: Numeric](sv: BV[T], sum: T): AliasTable[T] = {
    val used = sv.activeSize
    val table = new AliasTable[T](used)
    generateAlias(sv, sum, table)
  }

  def generateAlias[@specialized(Double, Int, Float, Long) T: ClassTag: Numeric](
    sv: BV[T], sum: T, table: AliasTable[T]): AliasTable[T] = {
    table.resetDist(sv, sum)
  }
}
