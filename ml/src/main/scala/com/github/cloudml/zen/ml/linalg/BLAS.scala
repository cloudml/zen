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

package com.github.cloudml.zen.ml.linalg

import com.github.cloudml.zen.ml.util.Logging
import com.github.fommil.netlib.{F2jBLAS, BLAS => NetlibBLAS}
import com.github.fommil.netlib.BLAS.{getInstance => NativeBLAS}
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector}

/**
 * Copy form MLlib
 */
private[ml] object BLAS extends Serializable with Logging {

  @transient private var _f2jBLAS: NetlibBLAS = _
  @transient private var _nativeBLAS: NetlibBLAS = _

  // For level-1 routines, we use Java implementation.
  private def f2jBLAS: NetlibBLAS = {
    if (_f2jBLAS == null) {
      _f2jBLAS = new F2jBLAS
    }
    _f2jBLAS
  }

  /**
   * y += a * x
   */
  def axpy(a: Double, x: Vector, y: Vector): Unit = {
    require(x.size == y.size)
    y match {
      case dy: DenseVector =>
        x match {
          case sx: SparseVector =>
            axpy(a, sx, dy)
          case dx: DenseVector =>
            axpy(a, dx, dy)
          case _ =>
            throw new UnsupportedOperationException(
              s"axpy doesn't support x type ${x.getClass}.")
        }
      case _ =>
        throw new IllegalArgumentException(
          s"axpy only supports adding to a dense vector but got type ${y.getClass}.")
    }
  }

  /**
   * y += a * x
   */
  private def axpy(a: Double, x: DenseVector, y: DenseVector): Unit = {
    val n = x.size
    f2jBLAS.daxpy(n, a, x.values, 1, y.values, 1)
  }

  /**
   * y += a * x
   */
  private def axpy(a: Double, x: SparseVector, y: DenseVector): Unit = {
    val xValues = x.values
    val xIndices = x.indices
    val yValues = y.values
    val nnz = xIndices.size

    if (a == 1.0) {
      var k = 0
      while (k < nnz) {
        yValues(xIndices(k)) += xValues(k)
        k += 1
      }
    } else {
      var k = 0
      while (k < nnz) {
        yValues(xIndices(k)) += a * xValues(k)
        k += 1
      }
    }
  }

  /**
   * dot(x, y)
   */
  def dot(x: Vector, y: Vector): Double = {
    require(x.size == y.size,
      "BLAS.dot(x: Vector, y:Vector) was given Vectors with non-matching sizes:" +
        " x.size = " + x.size + ", y.size = " + y.size)
    (x, y) match {
      case (dx: DenseVector, dy: DenseVector) =>
        dot(dx, dy)
      case (sx: SparseVector, dy: DenseVector) =>
        dot(sx, dy)
      case (dx: DenseVector, sy: SparseVector) =>
        dot(sy, dx)
      case (sx: SparseVector, sy: SparseVector) =>
        dot(sx, sy)
      case _ =>
        throw new IllegalArgumentException(s"dot doesn't support (${x.getClass}, ${y.getClass}).")
    }
  }

  /**
   * dot(x, y)
   */
  private def dot(x: DenseVector, y: DenseVector): Double = {
    val n = x.size
    f2jBLAS.ddot(n, x.values, 1, y.values, 1)
  }

  /**
   * dot(x, y)
   */
  private def dot(x: SparseVector, y: DenseVector): Double = {
    val xValues = x.values
    val xIndices = x.indices
    val yValues = y.values
    val nnz = xIndices.size

    var sum = 0.0
    var k = 0
    while (k < nnz) {
      sum += xValues(k) * yValues(xIndices(k))
      k += 1
    }
    sum
  }

  /**
   * dot(x, y)
   */
  private def dot(x: SparseVector, y: SparseVector): Double = {
    val xValues = x.values
    val xIndices = x.indices
    val yValues = y.values
    val yIndices = y.indices
    val nnzx = xIndices.size
    val nnzy = yIndices.size

    var kx = 0
    var ky = 0
    var sum = 0.0
    // y catching x
    while (kx < nnzx && ky < nnzy) {
      val ix = xIndices(kx)
      while (ky < nnzy && yIndices(ky) < ix) {
        ky += 1
      }
      if (ky < nnzy && yIndices(ky) == ix) {
        sum += xValues(kx) * yValues(ky)
        ky += 1
      }
      kx += 1
    }
    sum
  }

  /**
   * y = x
   */
  def copy(x: Vector, y: Vector): Unit = {
    val n = y.size
    require(x.size == n)
    y match {
      case dy: DenseVector =>
        x match {
          case sx: SparseVector =>
            val sxIndices = sx.indices
            val sxValues = sx.values
            val dyValues = dy.values
            val nnz = sxIndices.size

            var i = 0
            var k = 0
            while (k < nnz) {
              val j = sxIndices(k)
              while (i < j) {
                dyValues(i) = 0.0
                i += 1
              }
              dyValues(i) = sxValues(k)
              i += 1
              k += 1
            }
            while (i < n) {
              dyValues(i) = 0.0
              i += 1
            }
          case dx: DenseVector =>
            Array.copy(dx.values, 0, dy.values, 0, n)
        }
      case _ =>
        throw new IllegalArgumentException(s"y must be dense in copy but got ${y.getClass}")
    }
  }

  /**
   * x = a * x
   */
  def scal(a: Double, x: Vector): Unit = {
    x match {
      case sx: SparseVector =>
        f2jBLAS.dscal(sx.values.size, a, sx.values, 1)
      case dx: DenseVector =>
        f2jBLAS.dscal(dx.values.size, a, dx.values, 1)
      case _ =>
        throw new IllegalArgumentException(s"scal doesn't support vector type ${x.getClass}.")
    }
  }

  // For level-3 routines, we use the native BLAS.
  private def nativeBLAS: NetlibBLAS = {
    if (_nativeBLAS == null) {
      _nativeBLAS = NativeBLAS
    }
    _nativeBLAS
  }
}
