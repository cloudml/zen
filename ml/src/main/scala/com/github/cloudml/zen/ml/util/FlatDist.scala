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

import breeze.linalg.{SparseVector => brSV, DenseVector => brDV, sum, StorageVector}
import spire.math.{Numeric => spNum}


class FlatDist[@specialized(Double, Int, Float, Long) T: ClassTag](dataSize: Int,
  val isSparse: Boolean)
  (implicit ev: spNum[T]) extends DiscreteSampler[T] with Serializable {
  var _dist = initDist(dataSize)
  var _norm = ev.zero

  @inline def length: Int = _dist.length

  @inline def used: Int = _dist.activeSize

  @inline def norm: T = _norm

  def sampleRandom(gen: Random)(implicit gev: spNum[T]): Int = {
    val u = gen.nextDouble() * gev.toDouble(_norm)
    sampleFrom(gev.fromDouble(u), gen)
  }

  def sampleFrom(base: T, gen: Random): Int = {
    assert(ev.lt(base, _norm))
    val idx = if (used == 1) {
      0
    } else {
      var i = 0
      var cdf = ev.zero
      var found = false
      do {
        cdf = ev.plus(cdf, _dist(i))
        if (ev.lt(base, cdf)) {
          found = true
        } else {
          i += 1
        }
      } while (!found && i < used - 1)
      i
    }
    _dist.indexAt(idx)
  }

  def update(state: Int, value: => T): Unit = synchronized {
    val prev = _dist(state)
    _dist(state) = value
    val newNorm = ev.plus(_norm, ev.minus(value, prev))
    setNorm(newNorm)
  }

  def deltaUpdate(state: Int, delta: => T): Unit = synchronized {
    _dist(state) = ev.plus(_dist(state), delta)
    val newNorm = ev.plus(_norm, delta)
    setNorm(newNorm)
  }

  def resetDist(probs: Array[T], space: Array[Int], psize: Int): FlatDist[T] = synchronized {
    if (isSparse) {
      val dist = new brSV[T](space, probs, psize, probs.length)
      _dist = dist
      setNorm(sum(dist))
    } else {
      val dist = new brDV[T](probs)
      _dist = dist
      setNorm(sum(dist))
    }
  }

  def resetDist(distIter: Iterator[(Int, T)], psize: Int): FlatDist[T] = synchronized {
    reset(psize)
    var sum = ev.zero
    while (distIter.hasNext) {
      val (state, prob) = distIter.next()
      _dist(state) = prob
      sum = ev.plus(sum, prob)
    }
    setNorm(sum)
    this
  }

  private def reset(newSize: Int): FlatDist[T] = {
    _dist = initDist(newSize)
    _norm = ev.zero
    this
  }

  private def initDist(size: Int): StorageVector[T] = {
    if (isSparse) brSV.zeros[T](size) else brDV.zeros[T](size)
  }

  private def setNorm(norm: T): FlatDist[T] = {
    _norm = norm
    this
  }
}
