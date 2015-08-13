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
import com.github.cloudml.zen.ml.util.FTree._
import breeze.linalg.{Vector => BV, SparseVector => BSV, DenseVector => BDV}

private[zen] class FTree(dataSize: Int, val isSparse: Boolean)
  extends DiscreteSampler with Serializable {

  private var _regLen: Int = regularLen(dataSize)
  private var _tree: Array[Double] = new Array[Double](_regLen << 1)
  private var _space: Array[Int] = if (isSparse) null else new Array[Int](_regLen)
  private var _used: Int = dataSize

  def length: Int = _tree.length

  def size: Int = length

  def used: Int = _used

  def norm: Double = _tree(1)

  @inline private def leafOffset = _regLen

  @inline private def getLeaf(i: Int): Double = _tree(i + leafOffset)

  @inline private def setLeaf(i: Int, p: Double): Unit = {
    _tree(i + leafOffset) = p
  }

  /**
   * map pos in FTree to distribution state
   */
  private def toState(pos: Int): Int = {
    val i = pos - leafOffset
    if (isSparse) {
      i
    } else {
      _space(i)
    }
  }

  /**
   * map distribution state to pos in FTree
   */
  private def toTreePos(state: Int): Int = {
    val i = if (isSparse) {
      state
    } else {
      binarySearch(_space, state, 0, _used)
    }
    i + leafOffset
  }

  def sample(gen: Random): Int = {
    var u = gen.nextDouble() * _tree(1)
    var cur = 1
    while (cur < leafOffset) {
      val lc = cur << 1
      if (u < _tree(lc)) {
        cur = lc
      } else {
        u -= _tree(lc)
        cur = lc + 1
      }
    }
    toState(cur)
  }

  def update(state: Int, delta: Double): Unit = {
    var pos = toTreePos(state)
    if (pos < leafOffset) {
      pos = addState()
    }
    val p = _tree(pos)
    val np = p + delta
    if (np <= 0D) {
      delState(pos)
    } else {
      _tree(pos) = np
      updateAncestors(pos, delta)
    }
  }

  private def updateAncestors(cur: Int, delta: Double): Unit = {
    var pos = cur >> 1
    while (pos >= 1) {
      _tree(pos) += delta
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
      System.arraycopy(prevTree, prevRegLen, _tree, _regLen, prevUsed)
      buildFTree()
      if (isSparse) {
        System.arraycopy(prevSpace, 0, _space, 0, prevUsed)
      }
    } else {
      _used += 1
    }
    _used - 1 + leafOffset
  }

  private def delState(pos: Int): Unit = {
    val p = _tree(pos)
    _tree(pos) = 0D
    updateAncestors(pos, -p)
  }

  def resetDist(dist: BV[Double], sum: Double): this.type = {
    val used = dist.activeSize
    reset(used)
    dist match {
      case v: BDV[Double] =>
        assert(!isSparse)
        for ((i, prob) <- v.activeIterator) {
          setLeaf(i, prob)
        }
      case v: BSV[Double] =>
        assert(isSparse)
        for (((state, prob), i) <- v.activeIterator.zipWithIndex) {
          setLeaf(i, prob)
          _space(i) = state
        }
    }
    buildFTree()
    this
  }

  private def buildFTree(): this.type = {
    for (i <- leafOffset-1 to 1 by -1) {
      _tree(i) = _tree(i << 1) + _tree(i << 1 + 1)
    }
    this
  }

  private def reset(newDataSize: Int): this.type = {
    val regLen = regularLen(newDataSize)
    if (regLen > (_tree.length >> 1)) {
      _tree = new Array[Double](regLen << 1)
      _space = if (isSparse) null else new Array[Int](regLen)
    }
    _regLen = regLen
    _used = newDataSize
    for (i <- 0 until _regLen) {
      setLeaf(i, 0D)
    }
    _tree(1) = 0D
    this
  }
}

private[zen] object FTree {
  def generateFTree(sv: BV[Double]): FTree = {
    val used = sv.activeSize
    val ftree = sv match {
      case v: BDV[Double] => new FTree(used, isSparse=false)
      case v: BSV[Double] => new FTree(used, isSparse=true)
    }
    ftree.resetDist(sv, 0D)
  }

  private def regularLen(len: Int): Int = {
    if (len <= 0) {
      0
    } else {
      var lh = 1
      var lc = len
      while (lc != 1) {
        lh <<= 1
        lc >>= 1
      }
      if (lh == len) lh else lh << 1
    }
  }

  def binarySearch[T](arr: Array[T], key: T, start: Int, end: Int)(implicit num: Numeric[T]): Int = {
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
