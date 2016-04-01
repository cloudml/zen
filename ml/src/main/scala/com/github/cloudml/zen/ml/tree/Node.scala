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

import org.apache.spark.mllib.tree.model.{Node, Predict}

object Node {

  /**
    * Return a node with the given node id (but nothing else set).
    */
  def emptyNode(nodeIndex: Int): Node = new Node(nodeIndex, new Predict(Double.MinValue), -1.0,
    false, None, None, None, None)

  /**
    * Construct a node with nodeIndex, predict, impurity and isLeaf parameters.
    * This is used in `DecisionTree.findBestSplits` to construct child nodes
    * after finding the best splits for parent nodes.
    * Other fields are set at next level.
    *
    * @param nodeIndex integer node id, from 1
    * @param predict predicted value at the node
    * @param impurity current node impurity
    * @param isLeaf whether the node is a leaf
    * @return new node instance
    */
  def apply(
    nodeIndex: Int,
    predict: Predict,
    impurity: Double,
    isLeaf: Boolean): Node = {
    new Node(nodeIndex, predict, impurity, isLeaf, None, None, None, None)
  }

  /**
    * Return the index of the left child of this node.
    */
  def leftChildIndex(nodeIndex: Int): Int = nodeIndex << 1

  /**
    * Return the index of the right child of this node.
    */
  def rightChildIndex(nodeIndex: Int): Int = (nodeIndex << 1) + 1

  /**
    * Get the parent index of the given node, or 0 if it is the root.
    */
  def parentIndex(nodeIndex: Int): Int = nodeIndex >> 1

  /**
    * Return the level of a tree which the given node is in.
    */
  def indexToLevel(nodeIndex: Int): Int = if (nodeIndex == 0) {
    throw new IllegalArgumentException(s"0 is not a valid node index.")
  } else {
    java.lang.Integer.numberOfTrailingZeros(java.lang.Integer.highestOneBit(nodeIndex))
  }

  /**
    * Returns true if this is a left child.
    * Note: Returns false for the root.
    */
  def isLeftChild(nodeIndex: Int): Boolean = nodeIndex > 1 && nodeIndex % 2 == 0

  /**
    * Return the maximum number of nodes which can be in the given level of the tree.
    *
    * @param level  Level of tree (0 = root).
    */
  def maxNodesInLevel(level: Int): Int = 1 << level

  /**
    * Return the index of the first node in the given level.
    *
    * @param level  Level of tree (0 = root).
    */
  def startIndexInLevel(level: Int): Int = 1 << level

  /**
    * Traces down from a root node to get the node with the given node index.
    * This assumes the node exists.
    */
  def getNode(nodeIndex: Int, rootNode: Node): Node = {
    var tmpNode: Node = rootNode
    var levelsToGo = indexToLevel(nodeIndex)
    while (levelsToGo > 0) {
      if ((nodeIndex & (1 << levelsToGo - 1)) == 0) {
        tmpNode = tmpNode.leftNode.get
      } else {
        tmpNode = tmpNode.rightNode.get
      }
      levelsToGo -= 1
    }
    tmpNode
  }

}
