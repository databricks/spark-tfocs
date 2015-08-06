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
 * A trait for prox capable functions, which support computation of the proximity operator:
 *   prox_h(x, t) = argmin_z(h(z) + 0.5 * ||z - x||_2^2 / t)
 *
 * Both the minimizing z value and the function value h(z) may be computed, depending on the
 * mode specified.
 *
 * @tparam X A type representing a vector on which to evaluate the function.
 */
trait ProxCapableFunction[X] {
  /**
   * Evaluates prox_h at x with smoothing parameter t, returning both z and h(z) depending on the
   * mode specified.
   *
   * @param x The vector on which to evaluate the function.
   * @param t The smoothing parameter.
   * @param mode The computation mode. If mode.f is true, h(z) is returned. If mode.g is true, z
   *        is returned.
   * @return A Value containing h(z) and/or z, depending on the 'mode' parameter. The h(z) value
   *         is contained in value.f, while z is contained in value.g.
   */
  def apply(x: X, t: Double, mode: Mode): Value[X]

  /**
   * Evaluates prox_h at x with smoothing parameter t == 0.0, returning z.
   */
  def apply(x: X): Double = apply(x, 0.0, Mode(f = true, g = false)).f.get
}

/**
 * A function that returns constant zero.
 *
 * NOTE In matlab tfocs this functionality is implemented in prox_0.m.
 */
class ProxZeroVector extends ProxCapableFunction[Vector] {
  override def apply(x: Vector, t: Double, mode: Mode): Value[Vector] =
    Value(Some(0.0), Some(x))
}

/**
 * The proximity operator for the L1 norm.
 *
 * NOTE In matlab tfocs this functionality is implemented in prox_l1.m.
 */
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

/**
 * A projection onto the nonnegative orthant, implemented using an indicator function. The indicator
 * function returns 0 for values within the nonnegative orthant and Double.PositiveInfinity
 * otherwise.
 *
 * NOTE In matlab tfocs this functionality is implemented in proj_Rplus.m.
 */
class ProjRPlusVector extends ProxCapableFunction[Vector] {
  override def apply(x: Vector, t: Double, mode: Mode): Value[Vector] = {

    val g = if (mode.g) {
      Some(new DenseVector(x.toArray.map(math.max(_, 0.0))))
    } else {
      None
    }

    Value(Some(0.0), g)
  }

  override def apply(x: Vector): Double = if (x.toArray.min < 0.0) Double.PositiveInfinity else 0.0
}

/**
 * A projection onto a simple box defined by upper and lower limits on each vector element,
 * implemented using an indicator function. The indicator function returns 0 for values within the
 * box and Double.PositiveInfinity otherwise.
 *
 * NOTE In matlab tfocs this functionality is implemented in prox_box.m.
 */
class ProjBoxVector(l: Vector, u: Vector) extends ProxCapableFunction[Vector] {

  val limits = l.toArray.zip(u.toArray)

  override def apply(x: Vector, t: Double, mode: Mode): Value[Vector] = {

    val g = if (mode.g) {
      Some(new DenseVector(x.toArray.zip(limits).map(y =>
        // Bound each element using the lower and upper limit for that element.
        math.min(y._2._2, math.max(y._2._1, y._1)))))
    } else {
      None
    }

    Value(Some(0.0), g)
  }

  override def apply(x: Vector): Double = if (x.toArray.zip(limits).exists(y =>
    // If an element is outside of that element's bounds, return infinity.
    y._1 > y._2._2 || y._1 < y._2._1)) { Double.PositiveInfinity } else { 0.0 }
}
