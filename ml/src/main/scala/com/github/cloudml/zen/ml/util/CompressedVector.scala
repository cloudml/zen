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

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}
import me.lemire.integercompression.IntCompressor
import me.lemire.integercompression.differential.IntegratedIntCompressor


class CompressedVector(cdata: Array[Int],
  cindex: Array[Int]) extends Serializable {
  def toVector(numTopics: Int)(implicit nic: IntCompressor,
    iic: IntegratedIntCompressor): BV[Int] = {
    if (cindex == null) {
      new BDV(nic.uncompress(cdata))
    } else {
      new BSV(iic.uncompress(cindex), nic.uncompress(cdata), numTopics)
    }
  }
}

object CompressedVector {
  def fromVector(bv: BV[Int])(implicit nic: IntCompressor,
    iic: IntegratedIntCompressor): CompressedVector = bv match {
    case v: BDV[Int] =>
      new CompressedVector(nic.compress(v.data), null)
    case v: BSV[Int] =>
      new CompressedVector(nic.compress(v.data), iic.compress(v.index))
  }
}
