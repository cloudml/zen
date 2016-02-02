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

import spire.math.{Numeric => spNum}


class CompositeSampler(implicit ev: spNum[Double])
  extends Sampler[Double] {
  private var samplers: Seq[Sampler[_]] = _

  protected def numer: spNum[Double] = ev

  def norm: Double = {
    samplers.iterator.map(_.normDouble).sum
  }

  def sampleFrom(base: Double, gen: Random): Int = {
    val sampIter = samplers.iterator
    var curSampler = sampIter.next()
    var subNorm = curSampler.normDouble
    var remain = base
    var i = 0
    while (remain >= subNorm) {
      if (!sampIter.hasNext) {
        println(s"base=$base, norm=$norm, remain=$remain, i=$i/${samplers.length}, subNorm=$subNorm")
      }
      i += 1
      remain -= subNorm
      curSampler = sampIter.next()
      subNorm = curSampler.normDouble
    }
    curSampler.sampleFromDouble(remain, gen)
  }

  def resetComponents(samplers: Sampler[_]*): CompositeSampler = {
    this.samplers = samplers
    this
  }
}
