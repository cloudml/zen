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

import breeze.linalg.{Vector => BV, SparseVector => BSV, DenseVector => BDV}
import breeze.storage.Zero
import org.apache.spark.mllib.linalg.{DenseVector => SDV, Vector => SV, SparseVector => SSV}
import scala.language.implicitConversions
import scala.reflect.ClassTag

private[zen] object SparkUtils {
  implicit def toBreeze(sv: SV): BV[Double] = {
    sv match {
      case SDV(data) =>
        new BDV(data)
      case SSV(size, indices, values) =>
        new BSV(indices, values, size)
    }
  }

  implicit def fromBreeze(breezeVector: BV[Double]): SV = {
    breezeVector match {
      case v: BDV[Double] =>
        if (v.offset == 0 && v.stride == 1 && v.length == v.data.length) {
          new SDV(v.data)
        } else {
          new SDV(v.toArray) // Can't use underlying array directly, so make a new one
        }
      case v: BSV[Double] =>
        if (v.index.length == v.used) {
          new SSV(v.length, v.index, v.data)
        } else {
          new SSV(v.length, v.index.slice(0, v.used), v.data.slice(0, v.used))
        }
      case v: BV[_] =>
        sys.error("Unsupported Breeze vector type: " + v.getClass.getName)
    }
  }

  private def _conv[T1: ClassTag, T2: ClassTag](data: Array[T1]): Array[T2] = {
    data.map(_.asInstanceOf[T2]).array
  }

  def toBreezeConv[T: ClassTag](sv: SV): BV[T] = {
    implicit val conv: Array[Double] => Array[T] = _conv[Double, T]
    sv match {
      case SDV(data) =>
        new BDV[T](data)
      case SSV(size, indices, values) =>
        new BSV[T](indices, values, size)(Zero[T](0.asInstanceOf[T]))
    }
  }

  def fromBreezeConv[T: ClassTag](breezeVector: BV[T]): SV = {
    implicit val conv: Array[T] => Array[Double] = _conv[T, Double]
    breezeVector match {
      case v: BDV[T] =>
        if (v.offset == 0 && v.stride == 1 && v.length == v.data.length) {
          new SDV(v.data)
        } else {
          new SDV(v.toArray) // Can't use underlying array directly, so make a new one
        }
      case v: BSV[T] =>
        if (v.index.length == v.used) {
          new SSV(v.length, v.index, _conv[T, Double](v.data))
        } else {
          new SSV(v.length, v.index.slice(0, v.used), v.data.slice(0, v.used))
        }
      case v: BV[T] =>
        sys.error("Unsupported Breeze vector type: " + v.getClass.getName)
    }
  }
}
