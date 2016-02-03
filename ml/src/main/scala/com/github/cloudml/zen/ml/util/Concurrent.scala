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

import java.util.concurrent.{Executors, LinkedBlockingQueue, ThreadPoolExecutor}

import scala.concurrent._
import scala.concurrent.duration._


object Concurrent extends Serializable {
  @inline def withFuture[T](body: => T)(implicit es: ExecutionContextExecutorService): Future[T] = {
    Future(body)(es)
  }

  @inline def withAwaitReady[T](future: Future[T])(implicit es: ExecutionContextExecutorService): Unit = {
    Await.ready(future, 1.hour)
  }

  def withAwaitReadyAndClose[T](future: Future[T])(implicit es: ExecutionContextExecutorService): Unit = {
    Await.ready(future, 1.hour)
    closeExecutionContext(es)
  }

  @inline def withAwaitResult[T](future: Future[T])(implicit es: ExecutionContextExecutorService): T = {
    Await.result(future, 1.hour)
  }

  def withAwaitResultAndClose[T](future: Future[T])(implicit es: ExecutionContextExecutorService): T = {
    val res = Await.result(future, 1.hour)
    closeExecutionContext(es)
    res
  }

  @inline def initExecutionContext(numThreads: Int): ExecutionContextExecutorService = {
    ExecutionContext.fromExecutorService(Executors.newFixedThreadPool(numThreads))
  }

  @inline def closeExecutionContext(es: ExecutionContextExecutorService): Unit = {
    es.shutdown()
  }
}

object DebugConcurrent extends Serializable {
  def withFuture[T](body: => T)(implicit es: ExecutionContextExecutorService): Future[T] = {
    val future = Future(body)(es)
    future.onFailure { case e =>
      e.printStackTrace()
    }(scala.concurrent.ExecutionContext.Implicits.global)
    future
  }

  def withAwaitReady[T](future: Future[T])(implicit es: ExecutionContextExecutorService): Unit = {
    Await.ready(future, 1.hour)
  }

  def withAwaitReadyAndClose[T](future: Future[T])(implicit es: ExecutionContextExecutorService): Unit = {
    future.onComplete { _ =>
      closeExecutionContext(es)
    }(scala.concurrent.ExecutionContext.Implicits.global)
    Await.ready(future, 1.hour)
  }

  def withAwaitResult[T](future: Future[T])(implicit es: ExecutionContextExecutorService): T = {
    Await.result(future, 1.hour)
  }

  def withAwaitResultAndClose[T](future: Future[T])(implicit es: ExecutionContextExecutorService): T = {
    future.onComplete { _ =>
      closeExecutionContext(es)
    }(scala.concurrent.ExecutionContext.Implicits.global)
    Await.result(future, 1.hour)
  }

  def initExecutionContext(numThreads: Int): ExecutionContextExecutorService = {
    val es = new ThreadPoolExecutor(numThreads, numThreads, 0L, MILLISECONDS, new LinkedBlockingQueue[Runnable],
      Executors.defaultThreadFactory, new ThreadPoolExecutor.AbortPolicy)
    ExecutionContext.fromExecutorService(es)
  }

  def closeExecutionContext(es: ExecutionContextExecutorService): Unit = {
    es.shutdown()
    if (!es.awaitTermination(1L, SECONDS)) {
      System.err.println("Error: ExecutorService does not exit itself, force to terminate.")
    }
  }
}
