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

package com.github.cloudml.zen.ml.sampler

import java.util.Random
import scala.annotation.tailrec
import scala.reflect.ClassTag

import FTree._

import breeze.linalg.{SparseVector => brSV, DenseVector => brDV, Vector => brV}
import spire.math.{Numeric => spNum}


class FTree[@specialized(Double, Int, Float, Long) T: ClassTag](val isSparse: Boolean)
  (implicit ev: spNum[T]) extends DiscreteSampler[T] with Serializable {
  var _regLen: Int = _
  var _tree: Array[T] = _
  var _space: Array[Int] = _
  var _used: Int = _

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

  def sampleRandom(gen: Random)(implicit gev: spNum[T]): Int = {
    val u = gen.nextDouble() * gev.toDouble(_tree(1))
    sampleFrom(gev.fromDouble(u), gen)
  }

  def sampleFrom(base: T, gen: Random): Int = {
    assert(ev.lt(base, _tree(1)))
    if (_used == 1) {
      toState(1)
    } else {
      var u = base
      var cur = 1
      while (cur < leafOffset) {
        val lc = cur << 1
        val lcp = _tree(lc)
        if (ev.lt(u, lcp)) {
          cur = lc
        } else {
          u = ev.minus(u, lcp)
          cur = lc + 1
        }
      }
      toState(cur)
    }
  }

  def update(state: Int, value: => T): Unit = {
    assert(ev.lteqv(value, ev.zero))
    var pos = toTreePos(state)
    if (pos < leafOffset) {
      pos = addState()
    }
    if (ev.lteqv(value, ev.zero)) {
      delState(pos)
    } else {
      val p = _tree(pos)
      val delta = ev.minus(value, p)
      _tree(pos) = value
      updateAncestors(pos, delta)
    }
  }

  def deltaUpdate(state: Int, delta: => T): Unit = {
    var pos = toTreePos(state)
    if (pos < leafOffset) {
      pos = addState()
    }
    val p = _tree(pos)
    val np = ev.plus(p, delta)
    if (ev.lteqv(np, ev.zero)) {
      delState(pos)
    } else {
      _tree(pos) = np
      updateAncestors(pos, delta)
    }
  }

  private def updateAncestors(cur: Int, delta: T): Unit = {
    var pos = cur >> 1
    while (pos >= 1) {
      _tree(pos) = ev.plus(_tree(pos), delta)
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
    _tree(pos) = ev.zero
    updateAncestors(pos, ev.negate(p))
  }

  def resetDist(probs: Array[T], space: Array[Int], psize: Int): FTree[T] = {
    resetDist(space.iterator.zip(probs.iterator), psize)
  }

  def resetDist(distIter: Iterator[(Int, T)], psize: Int): FTree[T] = {
    reset(psize)
    if (!isSparse) {
      while (distIter.hasNext) {
        val (i, prob) = distIter.next()
        setLeaf(i, prob)
      }
    } else {
      var i = 0
      while (i < psize) {
        val (state, prob) = distIter.next()
        setLeaf(i, prob)
        _space(i) = state
        i += 1
      }
    }
    buildFTree()
    this
  }

  private def buildFTree(): FTree[T] = {
    var i = leafOffset - 1
    while (i >= 1) {
      _tree(i) = ev.plus(_tree(i << 1), _tree((i << 1) + 1))
      i -= 1
    }
    this
  }

  def reset(newSize: Int): FTree[T] = {
    _regLen = regularLen(newSize)
    if (_tree == null || _regLen > (_tree.length >> 1)) {
      _tree = new Array[T](_regLen << 1)
      _space = if (!isSparse) null else new Array[Int](_regLen)
    }
    _used = newSize
    var i = 0
    while (i < _regLen) {
      setLeaf(i, ev.zero)
      i += 1
    }
    _tree(1) = ev.zero
    this
  }
}

object FTree {
  def generateFTree[@specialized(Double, Int, Float, Long) T: ClassTag: spNum]
  (sv: brV[T]): FTree[T] = {
    val used = sv.activeSize
    val ftree = sv match {
      case v: brDV[T] => new FTree[T](isSparse=false)
      case v: brSV[T] => new FTree[T](isSparse=true)
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
    key: T, start: Int, end: Int)(implicit ev: spNum[T]): Int = {
    @tailrec def seg(s: Int, e: Int): Int = {
      if (s > e) return -1
      val mid = (s + e) >> 1
      mid match {
        case _ if ev.eqv(arr(mid), key) => mid
        case _ if ev.lt(arr(mid), key) => seg(mid + 1, e)
        case _ if ev.gt(arr(mid), key) => seg(s, mid - 1)
      }
    }
    seg(start, end - 1)
  }
}
