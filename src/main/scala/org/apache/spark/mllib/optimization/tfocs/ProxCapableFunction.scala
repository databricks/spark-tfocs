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

import org.apache.spark.mllib.linalg.{ DenseVector, Vector, Vectors }

/**
 * Trait for prox-capable functions.
 */
trait ProxCapableFunction[X] {
  /**
   * Evaluates this function at x with smoothing parameter t.
   */
  def apply(x: X, t: Double, mode: Mode): Value[Double, X]

  /**
   * Evaluates this function at x.
   */
  def apply(x: X): Double = apply(x, 0.0, Mode(f = true, g = false)).f.get
}

/** A function that always returns zero. */
class ZeroProxVector extends ProxCapableFunction[Vector] {
  override def apply(x: Vector, t: Double, mode: Mode): Value[Double, Vector] =
    Value(Some(0.0), Some(x))
}

/** A function that returns the L1 norm. */
class L1ProxVector(scale: Double) extends ProxCapableFunction[Vector] {
  override def apply(x: Vector, t: Double, mode: Mode): Value[Double, Vector] = {
    val shrinkage = scale * t
    val g = shrinkage match {
      case 0.0 => x
      case _ =>
        new DenseVector(x.toArray.map(x => x * (1.0 - math.min(shrinkage / math.abs(x), 1.0))))
    }
    val f = if (mode.f) Some(scale * Vectors.norm(g, 1)) else None
    Value(f, Some(g))
  }
}
