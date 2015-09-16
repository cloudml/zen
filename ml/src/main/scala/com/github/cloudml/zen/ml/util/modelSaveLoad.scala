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

import org.apache.hadoop.fs._
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.types.{DataType, StructField, StructType}
import org.json4s._
import org.json4s.jackson.JsonMethods._

import scala.reflect.ClassTag
import scala.reflect.runtime.universe.TypeTag

// copy form Spark MLlib
/**
 * Helper methods for loading models from files.
 */
private[ml] object LoaderUtils {

  /** Returns URI for path/data using the Hadoop filesystem */
  def dataPath(path: String): String = new Path(path, "data").toUri.toString

  /** Returns URI for path/metadata using the Hadoop filesystem */
  def metadataPath(path: String): String = new Path(path, "metadata").toUri.toString

  /**
   * Check the schema of loaded model data.
   *
   * This checks every field in the expected schema to make sure that a field with the same
   * name and DataType appears in the loaded schema.  Note that this does NOT check metadata
   * or containsNull.
   *
   * @param loadedSchema  Schema for model data loaded from file.
   * @tparam Data  Expected data type from which an expected schema can be derived.
   */
  def checkSchema[Data: TypeTag](loadedSchema: StructType): Unit = {
    // Check schema explicitly since erasure makes it hard to use match-case for checking.
    val expectedFields: Array[StructField] =
      ScalaReflection.schemaFor[Data].dataType.asInstanceOf[StructType].fields
    val loadedFields: Map[String, DataType] =
      loadedSchema.map(field => field.name -> field.dataType).toMap
    expectedFields.foreach { field =>
      assert(loadedFields.contains(field.name), s"Unable to parse model data." +
        s"  Expected field with name ${field.name} was missing in loaded schema:" +
        s" ${loadedFields.mkString(", ")}")
    }
  }

  /**
   * Load metadata from the given path.
   * @return (class name, version, metadata)
   */
  def loadMetadata(sc: SparkContext, path: String): (String, String, JValue) = {
    implicit val formats = DefaultFormats
    val metadata = parse(sc.textFile(metadataPath(path)).first())
    val clazz = (metadata \ "class").extract[String]
    val version = (metadata \ "version").extract[String]
    (clazz, version, metadata)
  }

  /**
   * Save an RDD to one HDFS file
   * @param sc SparkContext
   * @param rdd The RDD to save
   * @param outPathStr The HDFS file path of String
   * @param header Header line of HDFS file, used for storing some metadata
   * @param mapEle The function mapping each element of RDD to a line of String
   */
  def RDD2HDFSFile[T: ClassTag](sc: SparkContext,
    rdd: RDD[T],
    outPathStr: String,
    header: => String,
    mapEle: T => String): Unit = {
    val hdpconf = sc.hadoopConfiguration
    val fs = FileSystem.get(hdpconf)
    val outPath = new Path(outPathStr)
    if (fs.exists(outPath)) {
      throw new InvalidPathException(s"Output path $outPathStr already exists.")
    }
    val fout = fs.create(outPath)
    fout.write(header.getBytes)
    fout.write("\n".getBytes)
    rdd.toLocalIterator.foreach(e => {
      fout.write(mapEle(e).getBytes)
      fout.write("\n".getBytes)
    })
    fout.close()
  }

  /**
   * Load an RDD from one HDFS file
   * @param sc SparkContext
   * @param inPathStr The HDFS file path of String
   * @param init_f The function used for initialization after reading header
   * @param lineParser The function parses each line in HDFS file to an element of RDD
   */
  def HDFSFile2RDD[T: ClassTag, M: ClassTag](sc: SparkContext,
    inPathStr: String,
    init_f: String => M,
    lineParser: (M, String) => T): (M, RDD[T]) = {
    val rawrdd = sc.textFile(inPathStr)
    val header = rawrdd.first()
    val meta = init_f(header)
    val rdd: RDD[T] = rawrdd.mapPartitions(iter => {
      val first = iter.next()
      if (first == header) {
        iter
      } else {
        Iterator.single(first) ++ iter
      }
    }.map(lineParser(meta, _)))
    (meta, rdd)
  }
}
