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

import FTree._

import breeze.linalg.{Vector=>BV, SparseVector=>BSV, DenseVector=>BDV}


class FTree[@specialized(Double, Int, Float, Long) T: ClassTag](dataSize: Int,
  val isSparse: Boolean)
  (implicit num: Numeric[T]) extends DiscreteSampler[T] with Serializable {

  private var _regLen: Int = regularLen(dataSize)
  private var _tree: Array[T] = new Array[T](_regLen << 1)
  private var _space: Array[Int] = if (!isSparse) null else new Array[Int](_regLen)
  private var _used: Int = dataSize

  def length: Int = _tree.length

  def size: Int = length

  def used: Int = _used

  def norm: T = _tree(1)

  @inline private def leafOffset = _regLen

  @inline private def getLeaf(i: Int): T = _tree(i + leafOffset)

  @inline private def setLeaf(i: Int, p: T): Unit = {
    _tree(i + leafOffset) = p
  }

  /**
   * map pos in FTree to distribution state
   */
  private def toState(pos: Int): Int = {
    val i = pos - leafOffset
    if (!isSparse) {
      i
    } else {
      _space(i)
    }
  }

  /**
   * map distribution state to pos in FTree
   */
  private def toTreePos(state: Int): Int = {
    val i = if (!isSparse) {
      state
    } else {
      binarySearch(_space, state, 0, _used)
    }
    i + leafOffset
  }

  def sampleRandom(gen: Random): Int = {
    if (_used == 1) {
      toState(1)
    } else {
      var u = gen.nextDouble() * num.toDouble(_tree(1))
      var cur = 1
      while (cur < leafOffset) {
        val lc = cur << 1
        val lcp = num.toDouble(_tree(lc))
        if (u < lcp) {
          cur = lc
        } else {
          u -= lcp
          cur = lc + 1
        }
      }
      toState(cur)
    }
  }

  def sampleFrom(base: T, gen: Random): Int = {
    assert(num.lt(base, _tree(1)))
    if (_used == 1) {
      toState(1)
    } else {
      var u = base
      var cur = 1
      while (cur < leafOffset) {
        val lc = cur << 1
        val lcp = _tree(lc)
        if (num.lt(u, lcp)) {
          cur = lc
        } else {
          u = num.minus(u, lcp)
          cur = lc + 1
        }
      }
      toState(cur)
    }
  }

  def update(state: Int, value: T): Unit = synchronized {
    assert(num.lteq(value, num.zero))
    var pos = toTreePos(state)
    if (pos < leafOffset) {
      pos = addState()
    }
    if (num.lteq(value, num.zero)) {
      delState(pos)
    } else {
      val p = _tree(pos)
      val delta = num.minus(value, p)
      _tree(pos) = value
      updateAncestors(pos, delta)
    }
  }

  def deltaUpdate(state: Int, delta: T): Unit = synchronized {
    var pos = toTreePos(state)
    if (pos < leafOffset) {
      pos = addState()
    }
    val p = _tree(pos)
    val np = num.plus(p, delta)
    if (num.lteq(np, num.zero)) {
      delState(pos)
    } else {
      _tree(pos) = np
      updateAncestors(pos, delta)
    }
  }

  private def updateAncestors(cur: Int, delta: T): Unit = {
    var pos = cur >> 1
    while (pos >= 1) {
      _tree(pos) = num.plus(_tree(pos), delta)
      pos >>= 1
    }
  }

  private def addState(): Int = {
    if (_used == _regLen) {
      val prevRegLen = _regLen
      val prevTree = _tree
      val prevSpace = _space
      val prevUsed = _used
      reset(_used + 1)
      Array.copy(prevTree, prevRegLen, _tree, _regLen, prevUsed)
      buildFTree()
      if (isSparse) {
        Array.copy(prevSpace, 0, _space, 0, prevUsed)
      }
    } else {
      _used += 1
    }
    _used - 1 + leafOffset
  }

  private def delState(pos: Int): Unit = {
    val p = _tree(pos)
    _tree(pos) = num.zero
    updateAncestors(pos, num.negate(p))
  }

  def resetDist(distIter: Iterator[(Int, T)], used: Int): this.type = synchronized {
    reset(used)
    if (!isSparse) {
      for ((i, prob) <- distIter) {
        setLeaf(i, prob)
      }
    } else {
      for (((state, prob), i) <- distIter.zipWithIndex) {
        setLeaf(i, prob)
        _space(i) = state
      }
    }
    buildFTree()
    this
  }

  private def buildFTree(): this.type = {
    for (i <- leafOffset - 1 to 1 by -1) {
      _tree(i) = num.plus(_tree(i << 1), _tree((i << 1) + 1))
    }
    this
  }

  private def reset(newDataSize: Int): this.type = {
    val regLen = regularLen(newDataSize)
    if (regLen > (_tree.length >> 1)) {
      _tree = new Array[T](regLen << 1)
      _space = if (!isSparse) null else new Array[Int](regLen)
    }
    _regLen = regLen
    _used = newDataSize
    for (i <- 0 until _regLen) {
      setLeaf(i, num.zero)
    }
    _tree(1) = num.zero
    this
  }
}

object FTree {
  def generateFTree[@specialized(Double, Int, Float, Long) T: ClassTag: Numeric](sv: BV[T]): FTree[T] = {
    val used = sv.activeSize
    val ftree = sv match {
      case v: BDV[T] => new FTree[T](used, isSparse=false)
      case v: BSV[T] => new FTree[T](used, isSparse=true)
    }
    ftree.resetDist(sv.activeIterator, used)
  }

  private def regularLen(len: Int): Int = {
    require(len < (1<<30))
    // http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
    var v = len - 1
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    v += 1
    v
  }

  def binarySearch[@specialized(Double, Int, Float, Long) T](arr: Array[T],
    key: T, start: Int, end: Int)(implicit num: Numeric[T]): Int = {
    def seg(s: Int, e: Int): Int = {
      if (s > e) return -1
      val mid = (s + e) >> 1
      mid match {
        case _ if num.equiv(arr(mid), key) => mid
        case _ if num.lt(arr(mid), key) => seg(mid + 1, e)
        case _ if num.gt(arr(mid), key) => seg(s, mid - 1)
      }
    }
    seg(start, end - 1)
  }
}
