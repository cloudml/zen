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
import scala.reflect.ClassTag

import breeze.linalg.{SparseVector => brSV, DenseVector => brDV, StorageVector, Vector => brV}
import breeze.storage.Zero
import spire.math.{Numeric => spNum}


class FlatDist[@specialized(Double, Int, Float, Long) T: ClassTag](val isSparse: Boolean)
  (implicit ev: spNum[T]) extends DiscreteSampler[T] with Serializable {
  var _dist: StorageVector[T] = _
  var _norm: T = _

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
        cdf = ev.plus(cdf, _dist.valueAt(i))
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

  def update(state: Int, value: => T): Unit = {
    val prev = _dist(state)
    _dist(state) = value
    val newNorm = ev.plus(_norm, ev.minus(value, prev))
    setNorm(newNorm)
  }

  def deltaUpdate(state: Int, delta: => T): Unit = {
    _dist(state) = ev.plus(_dist(state), delta)
    val newNorm = ev.plus(_norm, delta)
    setNorm(newNorm)
  }

  def resetDist(probs: Array[T], space: Array[Int], psize: Int): FlatDist[T] = {
    implicit val zero = Zero(ev.zero)
    _dist = if (isSparse) {
      new brSV[T](space, probs, psize, probs.length)
    } else {
      new brDV[T](probs)
    }
    var sum = ev.zero
    var i = 0
    while (i < psize) {
      sum = ev.plus(sum, probs(i))
      i += 1
    }
    setNorm(sum)
  }

  def resetDist(distIter: Iterator[(Int, T)], psize: Int): FlatDist[T] = {
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

  def reset(newSize: Int): FlatDist[T] = {
    if (_dist == null || _dist.length < newSize) {
      implicit val zero = Zero(ev.zero)
      _dist = if (isSparse) brSV.zeros[T](newSize) else brDV.zeros[T](newSize)
    }
    _norm = ev.zero
    this
  }

  private def setNorm(norm: T): FlatDist[T] = {
    _norm = norm
    this
  }
}

object FlatDist {
  def generateFlat[@specialized(Double, Int, Float, Long) T: ClassTag: spNum]
  (sv: brV[T]): FlatDist[T] = {
    val used = sv.activeSize
    val flat = sv match {
      case v: brDV[T] => new FlatDist[T](isSparse=false)
      case v: brSV[T] => new FlatDist[T](isSparse=true)
    }
    flat.resetDist(sv.activeIterator, used)
  }
}
