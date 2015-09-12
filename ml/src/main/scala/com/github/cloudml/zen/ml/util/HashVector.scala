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

import breeze.linalg.{DenseVector => BDV}
import org.apache.spark.graphx2.util.collection.GraphXPrimitiveKeyOpenHashMap


class HashVector[@specialized(Double, Int, Float, Long) V: ClassTag](val _size: Int,
  val _hash: GraphXPrimitiveKeyOpenHashMap[Int, V])
  (implicit num: Numeric[V]) extends Serializable {
  require(_size > 0, "Vector size must be positive")

  def this(_size: Int)(implicit num: Numeric[V]) = this(_size, new GraphXPrimitiveKeyOpenHashMap(16))

  @inline def checkBound(i: Int): Unit = {
    if (i < 0 || i >= _size) throw new IndexOutOfBoundsException(s"Index out of range: $i")
  }

  @inline def checkVectorSize(b: HashVector[V]): Unit = {
    if (b._size > _size) throw new IndexOutOfBoundsException(s"Index out of range: ${b._size}")
  }

  def apply(i: Int): V = {
    checkBound(i)
    _hash.getOrElse(i, num.zero)
  }

  def update(i: Int, v: V): Unit = {
    checkBound(i)
    _hash(i) = v
  }

  @inline def length: Int = _size

  @inline def size: Int = length

  @inline def used: Int = _hash.keySet.size

  def add(i: Int, v: V): this.type = {
    checkBound(i)
    _hash.setMerge(i, v, num.plus)
    this
  }

  def minus(i: Int, v: V): this.type = {
    checkBound(i)
    _hash.setMerge(i, v, num.minus)
    this
  }

  @inline def add(iv: (Int, V)): this.type = add(iv._1, iv._2)

  @inline def minus(iv: (Int, V)): this.type = minus(iv._1, iv._2)

  def :+=(b: HashVector[V]): this.type = {
    checkVectorSize(b)
    b.activeIterator.foreach {
      case (i, v) => _hash.setMerge(i, v, num.plus)
    }
    this
  }

  def :+(b: HashVector[V]): HashVector[V] = {
    val hv = HashVector.zeros[V](_size)
    hv :+= this
    hv :+= b
    hv
  }

  def :-=(b: HashVector[V]): this.type = {
    checkVectorSize(b)
    b.activeIterator.foreach {
      case (i, v) => _hash.setMerge(i, v, num.minus)
    }
    this
  }

  def :-(b: HashVector[V]): HashVector[V] = {
    val hv = HashVector.zeros[V](_size)
    hv :+= this
    hv :-= b
    hv
  }

  @inline def +=(b: HashVector[V]): this.type = this :+= b

  @inline def +(b: HashVector[V]): HashVector[V] = this :+ b

  @inline def -=(b: HashVector[V]): this.type = this :-= b

  @inline def -(b: HashVector[V]): HashVector[V] = this :- b

  def :++=:(left: BDV[V]): BDV[V] = {
    for ((i, v) <- activeIterator) {
      left(i) = num.plus(left(i), v)
    }
    left
  }

  @inline def ++=:(left: BDV[V]): BDV[V] = left :++=: this

  @inline def activeSize: Int = used

  @inline def activeIterator: Iterator[(Int, V)] = _hash.iterator

  @inline def activeKeysIterator: Iterator[Int] = activeIterator.map(_._1)

  @inline def activeValuesIterator: Iterator[V] = activeIterator.map(_._2)

  def index: Array[Int] = activeKeysIterator.toArray

  def data: Array[V] = activeValuesIterator.toArray

  def mapValues[V2: ClassTag](f: V => V2)(implicit num: Numeric[V2]): HashVector[V2] = {
    val hv = HashVector.zeros[V2](_size)
    activeIterator.foreach {
      case (i, v) => hv(i) = f(v)
    }
    hv
  }

  def sum: V = activeValuesIterator.sum

  def norm: Double = math.sqrt(num.toDouble(activeValuesIterator.map(v => num.times(v, v)).sum))

  def distanceWith(b: HashVector[V]): Double = (this :- b).norm
}

object HashVector {
  def zeros[@specialized(Double, Int, Float, Long) V: ClassTag](size: Int)
    (implicit num: Numeric[V]): HashVector[V] = {
    new HashVector[V](size)
  }
}
