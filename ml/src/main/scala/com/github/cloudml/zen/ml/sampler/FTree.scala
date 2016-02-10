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

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}
import com.github.cloudml.zen.ml.sampler.FTree._
import spire.math.{Numeric => spNum}

import scala.collection.mutable
import scala.reflect.ClassTag


class FTree[@specialized(Double, Int, Float, Long) T: ClassTag](val isSparse: Boolean)
  (implicit ev: spNum[T]) extends DiscreteSampler[T] {
  var _tree: Array[T] = _
  var _space: Array[Int] = _
  var _index: mutable.HashMap[Int, Int] = _
  var _regUsed: Int = _
  var _used: Int = _

  protected def numer: spNum[T] = ev

  def length: Int = _tree.length

  def size: Int = length

  def used: Int = _used

  def norm: T = _tree(1)

  private def idx2State(i: Int): Int = {
    assert(i < _used)
    if (isSparse) _space(i) else i
  }

  private def state2Idx(state: Int): Int = {
    if (isSparse) _index.getOrElse(state, -1) else state
  }

  def sampleFrom(base: T, gen: Random): Int = {
    // assert(ev.lt(base, _tree(1)))
    if (_used == 1) {
      idx2State(0)
    } else {
      var u = base
      var pos = 1
      while (pos < _regUsed) {
        val lc = pos << 1
        val lcp = _tree(lc)
        if (ev.lt(u, lcp)) {
          pos = lc
        } else {
          u = ev.minus(u, lcp)
          pos = lc + 1
        }
      }
      idx2State(pos - _regUsed)
    }
  }

  def apply(state: Int): T = {
    val i = state2Idx(state)
    if (i < 0) {
      ev.zero
    } else {
      _tree(i + _regUsed)
    }
  }

  def update(state: Int, value: => T): Unit = {
    var i = state2Idx(state)
    if (i < 0) {
      i = addState(state)
    }
    if (ev.lteqv(value, ev.zero)) {
      delState(i)
    } else {
      val pos = i + _regUsed
      val p = _tree(pos)
      val delta = ev.minus(value, p)
      _tree(pos) = value
      updateAncestors(pos, delta)
    }
  }

  def deltaUpdate(state: Int, delta: => T): Unit = {
    var i = state2Idx(state)
    if (i < 0) {
      i = addState(state)
    }
    val pos = i + _regUsed
    val p = _tree(pos)
    val np = ev.plus(p, delta)
    if (ev.lteqv(np, ev.zero)) {
      delState(i)
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

  private def addState(state: Int): Int = {
    if (_used == _regUsed) {
      val newRegUsed = if (_regUsed == 0) 1 else _regUsed << 1
      Array.copy(_tree, _regUsed, _tree, newRegUsed, _used)
      _regUsed = newRegUsed
      zeroUsedPaddings()
      buildFTree()
    }
    val i = _used
    _space(i) = state
    _index(state) = i
    _used += 1
    i
  }

  private def delState(i: Int): Unit = {
    val pos = i + _regUsed
    val p = _tree(pos)
    _tree(pos) = ev.zero
    updateAncestors(pos, ev.negate(p))
  }

  def resetDist(probs: Array[T], space: Array[Int], psize: Int): FTree[T] = {
    reset(psize)
    setUsed(psize)
    var i = 0
    while (i < psize) {
      _tree(i + _regUsed) = probs(i)
      if (isSparse) {
        val state = space(i)
        _space(i) = state
        _index(state) = i
      }
      i += 1
    }
    zeroUsedPaddings()
    buildFTree()
  }

  def resetDist(distIter: Iterator[(Int, T)], psize: Int): FTree[T] = {
    val (space, probs) = distIter.toArray.unzip
    assert(probs.length == psize)
    resetDist(probs.toArray, space.toArray, psize)
  }

  private def buildFTree(): FTree[T] = {
    var i = _regUsed - 1
    while (i >= 1) {
      val lpi = i << 1
      _tree(i) = ev.plus(_tree(lpi), _tree(lpi + 1))
      i -= 1
    }
    this
  }

  def reset(capacity: Int): FTree[T] = {
    val regCap = regularLen(capacity)
    val fullLen = regCap << 1
    if (_tree == null) {
      _tree = new Array[T](fullLen)
      if (isSparse) {
        _space = new Array[Int](regCap)
        _index = new mutable.HashMap[Int, Int]()
      }
    } else {
      if (fullLen > _tree.length) {
        _tree = new Array[T](fullLen)
        if (isSparse) {
          _space = new Array[Int](regCap)
        }
      }
      _index.clear()
    }
    setUsed(0)
  }

  private def zeroUsedPaddings(): Unit = {
    val usedLen = _regUsed << 1
    var pos = _regUsed + _used
    while (pos < usedLen) {
      _tree(pos) = ev.zero
      pos += 1
    }
  }

  private def setUsed(used: Int): FTree[T] = {
    _used = used
    _regUsed = regularLen(_used)
    this
  }
}

object FTree {
  def generateFTree[@specialized(Double, Int, Float, Long) T: ClassTag: spNum]
  (sv: BV[T]): FTree[T] = {
    val used = sv.activeSize
    val ftree = sv match {
      case v: BDV[T] => new FTree[T](isSparse=false)
      case v: BSV[T] => new FTree[T](isSparse=true)
    }
    ftree.resetDist(sv.activeIterator, used)
  }

  def regularLen(len: Int): Int = {
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
}
