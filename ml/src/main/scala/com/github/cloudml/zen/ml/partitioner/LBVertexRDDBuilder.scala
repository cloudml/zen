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

package com.github.cloudml.zen.ml.partitioner

import scala.reflect.ClassTag

import org.apache.spark.graphx2._
import org.apache.spark.graphx2.impl._
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel


object LBVertexRDDBuilder {
  // TODO: correctly implement this instead of using HashPartitioner
  // Locality based VertexRDD Builder, make each vertex put in the partition where edges need it most.
  // But now GraphX's implementation of VertexRDD only support HashPartitioner.
  // To implement this, modify VertexRDD's code first, add PartitionID in RDD element.
  def fromEdgeRDD[VD: ClassTag, ED: ClassTag](edges: EdgeRDD[ED],
    storageLevel: StorageLevel): GraphImpl[VD, ED] = {
    val eimpl = edges.asInstanceOf[EdgeRDDImpl[ED, VD]]
    GraphImpl.fromEdgeRDD(eimpl, null.asInstanceOf[VD], storageLevel, storageLevel)
  }

  def fromEdges[VD: ClassTag, ED: ClassTag](edges: RDD[Edge[ED]],
    storageLevel: StorageLevel): GraphImpl[VD, ED] = {
    fromEdgeRDD[VD, ED](EdgeRDD.fromEdges[ED, VD](edges), storageLevel)
  }
}
