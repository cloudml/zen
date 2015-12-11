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

import breeze.linalg.{Vector => brV}
import com.github.fommil.netlib.BLAS.{getInstance => blas}
import spire.math.{Numeric => spNum}


class AliasTable[@specialized(Double, Int, Float, Long) T: ClassTag](implicit ev: spNum[T])
  extends DiscreteSampler[T] {
  var _l: Array[Int] = _
  var _h: Array[Int] = _
  var _p: Array[T] = _
  var _used: Int = _
  var _norm: T = _

  def used: Int = _used

  def length: Int = _l.length

  def size: Int = length

  def norm: T = _norm

  def sampleRandom(gen: Random)(implicit gev: spNum[T]): Int = {
    val u = gen.nextDouble() * gev.toDouble(_norm)
    sampleFrom(gev.fromDouble(u), gen)
  }

  def sampleFrom(base: T, gen: Random): Int = {
    // assert(ev.lt(base, _norm))
    if (_used == 1) {
      _l(0)
    } else {
      val bin = gen.nextInt(_used)
      val prob = _p(bin)
      if (ev.lt(base, prob)) {
        _l(bin)
      } else {
        _h(bin)
      }
    }
  }

  def update(state: Int, value: => T): Unit = {}

  def deltaUpdate(state: Int, delta: => T): Unit = {}

  def resetDist(probs: Array[T], space: Array[Int], psize: Int): AliasTable[T] = {
    // @inline def isClose(a: Double, b: Double): Boolean = math.abs(a - b) <= (1e-8 + math.abs(a) * 1e-6)
    reset(psize)
    var sum = ev.zero
    sum match {
      case _: Double =>
        val dprobs = probs.asInstanceOf[Array[Double]]
        val dscale = psize.toDouble
        val dsum = blas.dasum(psize, dprobs, 1)
        sum = ev.fromDouble(dsum)
        blas.dscal(psize, dscale, dprobs, 1)
      case _: Float =>
        val fprobs = probs.asInstanceOf[Array[Float]]
        val fscale = psize.toFloat
        val fsum = blas.sasum(psize, fprobs, 1)
        sum = ev.fromFloat(fsum)
        blas.sscal(psize, fscale, fprobs, 1)
      case _ =>
        val scale = ev.fromInt(psize)
        var i = 0
        while (i < psize) {
          val prob = probs(i)
          sum = ev.plus(sum, prob)
          probs(i) = ev.times(prob, scale)
          i += 1
        }
    }
    var ls = 0
    var le = 0
    var he = psize
    val sq = new Array[Int](psize)
    val pq = new Array[T](psize)
    var qi = 0
    var i = 0
    while (i < psize) {
      val prob = probs(i)
      val state = if (space == null) i else space(i)
      if (ev.lt(prob, sum)) {
        _l(le) = state
        _p(le) = prob
        le += 1
      } else {
        sq(qi) = state
        pq(qi) = prob
        qi += 1
      }
      i += 1
    }
    i = 0
    while (i < qi) {
      val state = sq(i)
      val prob = pq(i)
      if (ev.lt(prob, sum)) {
        _l(le) = state
        _p(le) = prob
        le += 1
        i += 1
      } else if (ls < le) {
        _h(ls) = state
        val split = _p(ls)
        ls += 1
        val prest = ev.minus(prob, ev.minus(sum, split))
        pq(i) = prest
      } else {
        // assert(isClose(ev.toDouble(prob), ev.toDouble(sum)))
        he -= 1
        _l(he) = state
        _h(he) = state
        i += 1
      }
    }
    // assert(le - ls <= 1 && math.abs(he - le) <= 1 && math.abs(he - ls) <= 1)
    setNorm(sum)
  }

  def resetDist(distIter: Iterator[(Int, T)], psize: Int): AliasTable[T] = {
    val (space, probs) = distIter.toArray.unzip
    assert(probs.length == psize)
    resetDist(probs.toArray, space.toArray, psize)
  }

  def reset(newSize: Int): AliasTable[T] = {
    if (_l == null || _l.length < newSize) {
      _l = new Array[Int](newSize)
      _h = new Array[Int](newSize)
      _p = new Array[T](newSize)
    }
    _used = newSize
    _norm = ev.zero
    this
  }

  private def setNorm(norm: T): AliasTable[T] = {
    _norm = norm
    this
  }
}

object AliasTable {
  def generateAlias[@specialized(Double, Int, Float, Long) T: ClassTag: spNum]
  (sv: brV[T]): AliasTable[T] = {
    val used = sv.activeSize
    val table = new AliasTable[T]
    table.resetDist(sv.activeIterator, used)
  }
}
