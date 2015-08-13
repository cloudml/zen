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
import breeze.linalg.{Vector => BV, SparseVector => BSV}

private[zen] class FTree(dataSize: Int, isSparse: Boolean)
  extends DiscreteSampler with Serializable {

  def length: Int = regularLen(dataSize)

  private var _tree: Array[Double] = new Array[Double](length << 1)
  private var _space: Array[Int] = if (isSparse) null else new Array[Int](dataSize)

  def used: Int = dataSize

  def norm: Double = _tree(1)

  private def reset(var dataSize, var isSparse): this.type = {
    _tree = tree
    _space = space
    _spc2idx = _space.zipWithIndex.toMap
    this
  }

  def update(state: Int, delta: Float): Unit = {
    _space.findAll(state).
    val regLen = _tree.length >> 1
    val i = _spc2idx(state)
    _tree(regLen + i) =
  }

  def sample(gen: Random): Int = {
    var u = gen.nextFloat() * _tree(1)
    val regLen = _tree.length >> 1
    var cur = 1
    while (cur < regLen) {
      val lc = cur << 1
      if (u < _tree(lc)) {
        cur = lc
      } else {
        u -= _tree(lc)
        cur = lc + 1
      }
    }
    _space(cur - regLen)
  }
}

private[zen] object FTree {
  def buildFTree(sv: BV[Float]): FTree = {
    val regLen = regularLen(sv.activeSize)
    val tree = Array.fill[Float](2 * regLen)(0.0f)
    val space = BSV.zeros[Int](sv.activeSize)
    for (((state, prob), i) <- sv.activeIterator.zipWithIndex) {
      tree(regLen + i) = prob
      space(regLen + i) = state
    }
    for (i <- regLen - 1 to 1 by -1) {
      tree(i) = tree(i << 1) + tree(i << 1 + 1)
    }
    (new FTree).setData(tree, space)
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
}
