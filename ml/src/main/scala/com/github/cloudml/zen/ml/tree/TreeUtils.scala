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

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkConf
import org.apache.spark.deploy.SparkHadoopUtil

object TreeUtils {
  def getFileSystem(conf: SparkConf, path: Path): FileSystem = {
    val hadoopConf = SparkHadoopUtil.get.newConfiguration(conf)
    if (sys.env.contains("HADOOP_CONF_DIR") || sys.env.contains("YARN_CONF_DIR")) {
      val hdfsConfPath = if (sys.env.get("HADOOP_CONF_DIR").isDefined) {
        sys.env.get("HADOOP_CONF_DIR").get + "/core-site.xml"
      } else {
        sys.env.get("YARN_CONF_DIR").get + "/core-site.xml"
      }
      hadoopConf.addResource(new Path(hdfsConfPath))
    }
    path.getFileSystem(hadoopConf)
  }

  def getPartitionOffsets(upper: Int, numPartitions: Int): (Array[Int], Array[Int]) = {
    val npp = upper / numPartitions
    val nppp = npp + 1
    val residual = upper - npp * numPartitions
    val boundary = residual * nppp
    val startPP = new Array[Int](numPartitions)
    val lcLenPP = new Array[Int](numPartitions)
    var i = 0
    while(i < numPartitions) {
      if (i < residual) {
        startPP(i) = nppp * i
        lcLenPP(i) = nppp
      }
      else{
        startPP(i) = boundary + (i - residual) * npp
        lcLenPP(i) = npp
      }
      i += 1
    }
    (startPP, lcLenPP)
  }
}
