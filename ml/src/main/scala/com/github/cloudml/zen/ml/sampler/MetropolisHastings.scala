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


class MetropolisHastings(implicit ev: spNum[Double])
  extends Sampler[Double] {
  type TransProb = Int => Double

  private var origFunc: TransProb = _
  private var proposal: Sampler[Double] = _
  private var state: Int = _

  protected def numer: spNum[Double] = ev

  def apply(state: Int): Double = origFunc(state)

  def norm: Double = proposal.norm

  def sampleFrom(base: Double, gen: Random): Int = {
    val newState = proposal.sampleFrom(base, gen)
    if (newState != state) {
      val ar = acceptRate(newState)
      if (ar >= 1.0 || gen.nextDouble() < ar) {
        state = newState
      }
    }
    state
  }

  private def acceptRate(newState:Int): Double = {
    origFunc(newState) * proposal(state) /
      (origFunc(state) * proposal(newState))
  }

  def resetProb(origFunc: TransProb,
    proposal: Sampler[Double],
    initState: Int): MetropolisHastings = {
    this.origFunc = origFunc
    this.proposal = proposal
    this.state = initState
    this
  }

  def resetProb(origFunc: TransProb,
    proposal: Sampler[Double],
    gen: Random): MetropolisHastings = {
    this.origFunc = origFunc
    this.proposal = proposal
    this.state = proposal.sampleRandom(gen)
    this
  }
}
