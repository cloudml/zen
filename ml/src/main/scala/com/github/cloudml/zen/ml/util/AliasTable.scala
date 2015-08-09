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
    if (prob > gen.nextFloat()) {
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
    val (lit, hit) = probs.map(t => (t._1, t._2 * used / sum)).partition(_._2 <= 1F)
    var lhead = 0
    var ltail = 0
    var htail = used
    def putPair: (Int, Float) => Unit = (t, pt) => {
      @inline def isClose: (Float, Float) => Boolean = (a, b) => abs(a - b) <= 1e-7 + abs(a) * 5e-4
      if (pt <= 0 || isClose(pt, 0F)) {
      } else if (isClose(pt, 1F)) {
        htail -= 1
        table.l(htail) = t
        table.h(htail) = t
      } else if (pt < 1F) {
        table.l(ltail) = t
        table.p(ltail) = pt
        ltail += 1
      } else {
        val pd = if (lhead == ltail) {  // no to-be-filled bucket
          htail -= 1
          table.l(htail) = t
          table.h(htail) = t
          pt - 1F
        } else {  // first tbf bucket
          table.h(lhead) = t
          val pl = table.p(lhead)
          lhead += 1
          pl + pt - 1F
        }
        putPair(t, pd)
      }
    }
    lit.foreach(t => putPair(t._1, t._2))
    hit.foreach(t => putPair(t._1, t._2))
    assert(lhead == ltail && ltail == htail ||  // normal end
      lhead == ltail - 1 && ltail == htail ||  // last pt=1F saved as pt<1F
      lhead == ltail - 1 && lhead == htail)  // last small fraction left
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
