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

package com.github.cloudml.zen.ml.tree


object ProbabilityFunctions{
  // probit function
  val ProbA = Array(3.3871328727963666080e0, 1.3314166789178437745e+2, 1.9715909503065514427e+3,
    1.3731693765509461125e+4, 4.5921953931549871457e+4, 6.7265770927008700853e+4, 3.3430575583588128105e+4,
    2.5090809287301226727e+3)
  val ProbB = Array(4.2313330701600911252e+1, 6.8718700749205790830e+2, 5.3941960214247511077e+3,
    2.1213794301586595867e+4, 3.9307895800092710610e+4, 2.8729085735721942674e+4, 5.2264952788528545610e+3)

  val ProbC = Array(1.42343711074968357734e0, 4.63033784615654529590e0, 5.76949722146069140550e0,
    3.64784832476320460504e0, 1.27045825245236838258e0, 2.41780725177450611770e-1, 2.27238449892691845833e-2,
    7.74545014278341407640e-4)
  val ProbD = Array(2.05319162663775882187e0, 1.67638483018380384940e0, 6.89767334985100004550e-1,
    1.48103976427480074590e-1, 1.51986665636164571966e-2, 5.47593808499534494600e-4, 1.05075007164441684324e-9)

  val ProbE = Array(6.65790464350110377720e0, 5.46378491116411436990e0, 1.78482653991729133580e0,
    2.96560571828504891230e-1, 2.65321895265761230930e-2, 1.24266094738807843860e-3, 2.71155556874348757815e-5,
    2.01033439929228813265e-7)
  val ProbF = Array(5.99832206555887937690e-1, 1.36929880922735805310e-1, 1.48753612908506148525e-2,
    7.86869131145613259100e-4, 1.84631831751005468180e-5, 1.42151175831644588870e-7, 2.04426310338993978564e-15)

  def Probit(p: Double): Double ={
    val q = p - 0.5
    var r = 0.0
    if (math.abs(q) < 0.425) {
      r = 0.180625 - q * q
      q * coeff(ProbA, ProbB, r)
    } else {
      r = if (q < 0) p else 1 - p
      r = math.sqrt(-math.log(r))
      var retval = 0.0
      if(r < 5) {
        r = r - 1.6
        retval = coeff(ProbC, ProbD, r)
      } else {
        r = r - 5
        retval = coeff(ProbE, ProbF, r)
      }
      if (q >= 0) retval else -retval
    }
  }

  def coeff(p1: Array[Double], p2: Array[Double], r: Double): Double = {
    (((((((p1(7) * r + p1(6)) * r + p1(5)) * r + p1(4)) * r + p1(3)) * r + p1(2)) * r + p1(1)) * r + p1(0)) /
    (((((((p2(6) * r + p2(5)) * r + p2(4)) * r + p2(3)) * r + p2(2)) * r + p2(1)) * r + p2(0)) * r + 1.0)
  }

  // The approximate complimentary error function (i.e., 1-erf).
  def erfc(x: Double): Double = {
    if (x.isInfinity) {
      if(x.isPosInfinity) 1.0 else -1.0
    } else {
      val p = 0.3275911
      val a1 = 0.254829592
      val a2 = -0.284496736
      val a3 = 1.421413741
      val a4 = -1.453152027
      val a5 = 1.061405429

      val t = 1.0 / (1.0 + p * math.abs(x))
      val ev = ((((((((a5 * t) + a4) * t) + a3) * t) + a2) * t + a1) * t) * scala.math.exp(-(x * x))
      if (x >= 0) ev else 2 - ev
    }
  }
}
