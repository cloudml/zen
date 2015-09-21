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

class CompactVector[@specialized(Int, Long, Float, Double) T](var data: Array[T],
  var gapBase: Array[Int],
  var offset: Array[Int],
  val size: Int) extends Serializable {

  def checkBound(i: Int): Unit = {
    if (i < 0 || i >= size) throw new ArrayIndexOutOfBoundsException(i)
  }

  def apply(i: Int): T = {
    checkBound(i)
    data(i)
  }

  def update(i: Int, v: T): Unit = {
    checkBound(i)
    data(i) = v
  }
}
