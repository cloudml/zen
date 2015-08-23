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

import scala.reflect.ClassTag
import breeze.collection.mutable.OpenAddressHashArray
import breeze.linalg.{Vector => BV, DenseVector=>BDV, HashVector=>BHV}
import breeze.storage.Zero


class HashVector[@specialized(Double, Int, Float, Long) T: ClassTag](
  val ha: OpenAddressHashArray[T])(implicit num: Numeric[T])
  extends BV[T] with Serializable {

  def apply(i: Int): T = ha(i)

  def update(i: Int, v: T): Unit = {
    ha(i) = v
  }

  def length: Int = ha.length

  @inline def used: Int = ha.activeSize

  def add(i: Int, v: T): this.type = {
    ha(i) = num.plus(ha(i), v)
    this
  }

  @inline def add(iv: (Int, T)): this.type = add(iv._1, iv._2)

  def :+=(b: HashVector[T]): this.type = {
    b.activeIterator.foreach(add)
    this
  }

  def :+(b: HashVector[T]): HashVector[T] = {
    val t = HashVector.zeros[T](length)
    t :+= this
    t :+= b
    t
  }

  def :++=:(left: BDV[T]): BDV[T] = {
    for ((i, v) <- activeIterator) {
      left(i) = num.plus(left(i), v)
    }
    left
  }

  @inline def +=(b: HashVector[T]): this.type = this :+= b

  @inline def +(b: HashVector[T]): HashVector[T] = this :+ b

  @inline def ++=:(left: BDV[T]): BDV[T] = left :++=: this

  def activeSize: Int = ha.activeSize

  def activeIterator: Iterator[(Int, T)] = ha.activeIterator

  def activeKeysIterator: Iterator[Int] = ha.activeKeysIterator

  def activeValuesIterator: Iterator[T] = ha.valuesIterator

  def toBHV(implicit num: Numeric[T]): BHV[T] = {
    implicit val zero = Zero(num.zero)
    val bhv = BHV.zeros[T](size)
    for ((i, v) <- activeIterator) {
      bhv(i) = v
    }
    bhv
  }

  def repr = this

  def copy: HashVector[T] = new HashVector[T](ha.copy)
}

object HashVector {
  def zeros[@specialized(Double, Int, Float, Long) T: ClassTag](
    size: Int)(implicit num: Numeric[T]): HashVector[T] = {
    implicit val zero = Zero(num.zero)
    new HashVector[T](new OpenAddressHashArray[T](size))
  }
}
