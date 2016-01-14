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

import java.lang.ref.WeakReference
import java.util.{TimerTask, Timer}
import org.apache.spark.Logging

private[zen] object SparkHacker extends Logging {

  /** Run GC and make sure it actually has run */
  def runGC() {
    val weakRef = new WeakReference(new Object())
    val startTime = System.currentTimeMillis
    System.gc() // Make a best effort to run the garbage collection. It *usually* runs GC.
    // Wait until a weak reference object has been GCed
    System.runFinalization()
    while (weakRef.get != null) {
      System.gc()
      System.runFinalization()
      Thread.sleep(200)
      if (System.currentTimeMillis - startTime > 30000) {
        throw new Exception("automatically cleanup error")
      }
    }
  }

  def gcCleaner(delaydSeconds: Int, periodSeconds: Int, tag: String) {
    val timer = new Timer(tag + " cleanup timer", true)
    val task = new TimerTask {
      override def run() {
        try {
          runGC
          logInfo("Ran metadata cleaner for " + tag)
        } catch {
          case e: Exception => logError("Error running cleanup task for " + tag, e)
        }
      }
    }
    timer.schedule(task, delaydSeconds * 1000, periodSeconds * 1000)
  }
}
