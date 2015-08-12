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
import math.abs
import breeze.linalg.{Vector => BV, sum => brzSum}

private[zen] class AliasTable(initUsed: Int) extends Serializable {

  private var _l: Array[Int] = new Array[Int](initUsed)
  private var _h: Array[Int] = new Array[Int](initUsed)
  private var _p: Array[Double] = new Array[Double](initUsed)
  private var _used = initUsed
  private var _norm = 0D

  def l: Array[Int] = _l

  def h: Array[Int] = _h

  def p: Array[Double] = _p

  def used: Int = _used

  def length: Int = size

  def size: Int = l.length

  def norm: Double = _norm

  def sampleAlias(gen: Random): Int = {
    val bin = gen.nextInt(_used)
    val prob = _p(bin)
    if (gen.nextDouble() * _norm < prob) {
      _l(bin)
    } else {
      _h(bin)
    }
  }

  private[AliasTable] def reset(newSize: Int): this.type = {
    if (_l.length < newSize) {
      _l = new Array[Int](newSize)
      _h = new Array[Int](newSize)
      _p = new Array[Double](newSize)
    }
    _used = newSize
    _norm = 0D
    this
  }

  private[AliasTable] def setNorm(norm: Double): this.type = {
    _norm = norm
    this
  }
}

private[zen] object AliasTable {
  type Pair = (Int, Double)

  @transient private lazy val tableOrdering = new scala.math.Ordering[Pair] {
    override def compare(x: Pair, y: Pair): Int = {
      Ordering[Double].compare(x._2, y._2)
    }
  }
  @transient private lazy val tableReverseOrdering = tableOrdering.reverse

  def generateAlias(sv: BV[Double]): AliasTable = {
    val used = sv.activeSize
    val sum = brzSum(sv)
    val probs = sv.activeIterator.slice(0, used)
    generateAlias(probs, sum, used)
  }

  def generateAlias(probs: Iterator[Pair], sum: Double, used: Int): AliasTable = {
    val table = new AliasTable(used)
    generateAlias(probs, sum, used, table)
  }

  def generateAlias(
    probs: Iterator[Pair],
    sum: Double,
    used: Int,
    table: AliasTable): AliasTable = {
    table.reset(used)
    val (loList, hiList) = probs.map(t => (t._1, t._2 * used)).toList.partition(_._2 < sum)
    var ls = 0
    var le = 0
    var end = used
    @inline def isClose(a: Double, b: Double): Boolean = abs(a - b) <= (1e-8 + abs(a) * 1e-6)
    def putAlias(list: List[Pair], rest: List[Pair]): List[Pair] = list match {
      case Nil => rest
      case (t, pt) :: rlist if pt < sum =>
        table.l(le) = t
        table.p(le) = pt
        le += 1
        putAlias(rlist, rest)
      case (t, pt) :: rlist if ls < le =>
        table.h(ls) = t
        val pl = table.p(ls)
        ls += 1
        val pd = pt - (sum - pl)
        putAlias(List((t, pd)) ++ rlist, rest)
      case (t, pt) :: rlist=>
        putAlias(rlist, rest ++ List((t, pt)))
    }
    def putRest(rest: List[Pair]): Unit = rest match {
      case Nil => Unit
      case (t, pt) :: rrest =>
        assert(isClose(pt, sum))
        end -= 1
        table.l(end) = t
        table.h(end) = t
        putRest(rrest)
    }
    putRest(putAlias(hiList, putAlias(loList, List())))
    assert(ls == le && end == ls || ls == le - 1 && (end == ls || end == le))
    table.setNorm(sum)
    table
  }

  def generateAlias(sv: BV[Double], sum: Double): AliasTable = {
    val used = sv.activeSize
    val probs = sv.activeIterator.slice(0, used)
    generateAlias(probs, sum, used)
  }

  def generateAlias(sv: BV[Double], sum: Double, table: AliasTable): AliasTable = {
    val used = sv.activeSize
    val probs = sv.activeIterator.slice(0, used)
    generateAlias(probs, sum, used, table)
  }
}
