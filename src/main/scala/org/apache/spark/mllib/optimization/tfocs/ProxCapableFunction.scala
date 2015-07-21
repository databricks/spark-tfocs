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
 *
 * @tparam X Type representing a vector on which to evaluate the function.
 */
trait ProxCapableFunction[X] {
  /**
   * Evaluates this function at x with smoothing parameter t, returning the minimum value and
   * minimizing vector.
   */
  def apply(x: X, t: Double, mode: Mode): Value[X]

  /**
   * Evaluates this function at x.
   */
  def apply(x: X): Double = apply(x, 0.0, Mode(f = true, g = false)).f.get
}

/** A function that always returns zero. */
class ProxZeroVector extends ProxCapableFunction[Vector] {
  override def apply(x: Vector, t: Double, mode: Mode): Value[Vector] =
    Value(Some(0.0), Some(x))
}

/** A function that returns the L1 norm. */
class ProxL1Vector(scale: Double) extends ProxCapableFunction[Vector] {
  override def apply(x: Vector, t: Double, mode: Mode): Value[Vector] = {
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

/** A function that projects onto the positive orthant. */
class ProjRPlusVector extends ProxCapableFunction[Vector] {
  override def apply(x: Vector, t: Double, mode: Mode): Value[Vector] = {

    val g = if (mode.g) {
      Some(new DenseVector(x.toArray.map(math.max(_, 0.0))))
    } else {
      None
    }

    Value(Some(0.0), g)
  }
}

/**
 * A function that projects onto a simple box defined by upper and lower limits on each vector
 * element.
 */
class ProjBoxVector(l: Vector, u: Vector) extends ProxCapableFunction[Vector] {

  val limits = l.toArray.zip(u.toArray)

  override def apply(x: Vector, t: Double, mode: Mode): Value[Vector] = {

    val g = if (mode.g) {
      Some(new DenseVector(x.toArray.zip(limits).map(y =>
        math.min(y._2._2, math.max(y._1, y._2._1)))))
    } else {
      None
    }

    Value(Some(0.0), g)
  }
}
