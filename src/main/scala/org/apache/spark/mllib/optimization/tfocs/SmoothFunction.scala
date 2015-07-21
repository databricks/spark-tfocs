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

package org.apache.spark.mllib.optimization.tfocs

import org.apache.spark.mllib.linalg.{ DenseVector, Vectors, Vector }
import org.apache.spark.mllib.linalg.BLAS
import org.apache.spark.mllib.optimization.tfocs.CheckedIterator._
import org.apache.spark.rdd.RDD

/**
 * Trait for smooth functions.
 *
 * @tparam X Type representing a vector on which to evaluate the function.
 */
trait SmoothFunction[X] {
  /**
   * Evaluates this function at x and returns the function value and its gradient based on the mode
   * specified.
   */
  def apply(x: X, mode: Mode): Value[X]

  /**
   * Evaluates this function at x.
   */
  def apply(x: X): Double = apply(x, Mode(f = true, g = false)).f.get
}

/** The squared error function applied to RDD[Double] vectors, with a constant factor of 0.5. */
class SmoothQuadRDDDouble(x0: RDD[Double]) extends SmoothFunction[RDD[Double]] {

  x0.cache()

  override def apply(x: RDD[Double], mode: Mode): Value[RDD[Double]] = {
    val g = x.zip(x0).map(y => y._1 - y._2)
    if (mode.f && mode.g) g.cache()
    val f = if (mode.f) Some(g.treeAggregate(0.0)((sum, y) => sum + y * y, _ + _) / 2.0) else None
    Value(f, Some(g))
  }
}

/** The squared error function applied to RDD[Vector] vectors, with a constant factor of 0.5. */
class SmoothQuadRDDVector(x0: RDD[Vector]) extends SmoothFunction[RDD[Vector]] {

  x0.cache()

  override def apply(x: RDD[Vector], mode: Mode): Value[RDD[Vector]] = {
    val g = x.zip(x0).map(y =>
      new DenseVector(y._1.toArray.toIterator.checkedZip(y._2.toArray.toIterator).map(z =>
        z._1 - z._2).toArray): Vector)
    if (mode.f && mode.g) g.cache()
    val f = if (mode.f) {
      Some(g.treeAggregate(0.0)((sum, x) => sum + Math.pow(Vectors.norm(x, 2), 2), _ + _) / 2.0)
    } else {
      None
    }
    Value(f, Some(g))
  }
}

