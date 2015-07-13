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
import org.apache.spark.mllib.linalg.{DenseVector => SDV, Vector => SV, SparseVector => SSV}
import scala.language.implicitConversions

private[zen] object SparkUtils {
  implicit def toBreeze[T](sv: SV): BV[T] = {
    sv match {
      case SDV(data) =>
        new BDV[T](data.map(_.asInstanceOf[T]))
      case SSV(size, indices, values) =>
        new BSV[T](indices, values.map(_.asInstanceOf[T]), size)
    }
  }

  implicit def fromBreeze[T](breezeVector: BV[T]): SV = {
    breezeVector match {
      case v: BDV[_] =>
        if (v.offset == 0 && v.stride == 1 && v.length == v.data.length) {
          new SDV(v.data.map(_.asInstanceOf[Double]))
        } else {
          new SDV(v.toArray.map(_.asInstanceOf[Double])) // Can't use underlying array directly, so make a new one
        }
      case v: BSV[_] =>
        if (v.index.length == v.used) {
          new SSV(v.length, v.index, v.data.map(_.asInstanceOf[Double]))
        } else {
          new SSV(v.length, v.index.slice(0, v.used), v.data.slice(0, v.used).map(_.asInstanceOf[Double]))
        }
      case v: BV[_] =>
        sys.error("Unsupported Breeze vector type: " + v.getClass.getName)
    }
  }
}
