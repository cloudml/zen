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

import java.util.{PriorityQueue => JPriorityQueue, Random}

import breeze.linalg.{Vector => BV, sum => brzSum}

private[zen] class AliasTable(initUsed: Int) extends Serializable {

  private var _l: Array[Int] = new Array[Int](initUsed)
  private var _h: Array[Int] = new Array[Int](initUsed)
  private var _p: Array[Float] = new Array[Float](initUsed)
  private var _used = initUsed

  def l: Array[Int] = _l

  def h: Array[Int] = _h

  def p: Array[Float] = _p

  def used: Int = _used

  def length: Int = size

  def size: Int = l.length

  def sampleAlias(gen: Random): Int = {
    val bin = gen.nextInt(_used)
    val prob = _p(bin)
    if (_used * prob > gen.nextFloat()) {
      _l(bin)
    } else {
      _h(bin)
    }
  }

  private[AliasTable] def reset(newSize: Int): this.type = {
    if (_l.length < newSize) {
      _l = new Array[Int](newSize)
      _h = new Array[Int](newSize)
      _p = new Array[Float](newSize)
    }
    _used = newSize
    this
  }
}

private[zen] object AliasTable {
  @transient private lazy val tableOrdering = new scala.math.Ordering[(Any, Float)] {
    override def compare(x: (Any, Float), y: (Any, Float)): Int = {
      Ordering[Float].compare(x._2, y._2)
    }
  }
  @transient private lazy val tableReverseOrdering = tableOrdering.reverse

  def generateAlias(sv: BV[Float]): AliasTable = {
    val used = sv.activeSize
    val sum = brzSum(sv)
    val probs = sv.activeIterator.slice(0, used)
    generateAlias(probs, sum, used)
  }

  def generateAlias(probs: Iterator[(Int, Float)], sum: Float, used: Int): AliasTable = {
    val table = new AliasTable(used)
    generateAlias(probs, sum, used, table)
  }

  def generateAlias(
    probs: Iterator[(Int, Float)],
    sum: Float,
    used: Int,
    table: AliasTable): AliasTable = {
    table.reset(used)
    val pMean = 1.0f / used
    val lq = new JPriorityQueue[(Int, Float)](used, tableOrdering)
    val hq = new JPriorityQueue[(Int, Float)](used, tableReverseOrdering)

    probs.slice(0, used).foreach { pair =>
      val i = pair._1
      val pi = pair._2 / sum
      if (pi < pMean) {
        lq.add((i, pi))
      } else {
        hq.add((i, pi))
      }
    }

    var offset = 0
    while (!lq.isEmpty & !hq.isEmpty) {
      val (i, pi) = lq.remove()
      val (h, ph) = hq.remove()
      table.l(offset) = i
      table.h(offset) = h
      table.p(offset) = pi
      val pd = ph - (pMean - pi)
      if (pd >= pMean) {
        hq.add((h, pd))
      } else {
        lq.add((h, pd))
      }
      offset += 1
    }
    while (!hq.isEmpty) {
      val (h, ph) = hq.remove()
      table.l(offset) = h
      table.h(offset) = h
      table.p(offset) = ph
      offset += 1
    }

    while (!lq.isEmpty) {
      val (i, pi) = lq.remove()
      table.l(offset) = i
      table.h(offset) = i
      table.p(offset) = pi
      offset += 1
    }
    table
  }

  def generateAlias(sv: BV[Float], sum: Float): AliasTable = {
    val used = sv.activeSize
    val probs = sv.activeIterator.slice(0, used)
    generateAlias(probs, sum, used)
  }

  def generateAlias(sv: BV[Float], sum: Float, table: AliasTable): AliasTable = {
    val used = sv.activeSize
    val probs = sv.activeIterator.slice(0, used)
    generateAlias(probs, sum, used, table)
  }
}
