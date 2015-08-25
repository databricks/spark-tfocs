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
 * A trait for prox capable functions which support efficient proximity minimization, as expressed
 * by the proximity operator:
 *   x = prox_h(z, t) = argmin_x(h(x) + 0.5 * ||x - z||_2^2 / t)
 *
 * Both the minimizing x value and the function value h(x) may be computed, depending on the
 * mode specified.
 *
 * @tparam X A type representing a vector on which to evaluate the function.
 */
trait ProxCapableFunction[X] {

  /**
   * Evaluate the proximity operator prox_h at z with parameter t, returning both x and h(x)
   * depending on the mode specified.
   *
   * @param z The vector on which to evaluate the proximity operator.
   * @param t The proximity parameter.
   * @param mode The computation mode. If mode.f is true, h(x) is returned. If mode.minimizer is
   *        true, x is returned.
   *
   * @return A Value containing x, the vector minimizing the proximity function prox_h, and/or h(x),
   *         the function value at x. The exact list of values computed and returned depends on the
   *         attributes of the supplied 'mode' parameter. The returned Value contains h(x) in its
   *         'f' attribute, while x is contained in the 'minimizer' attribute.
   */
  def apply(z: X, t: Double, mode: ProxMode): ProxValue[X]

  /** Evaluate the function h(x) at x. Does not perform proximity minimization. */
  def apply(x: X): Double
}

/**
 * The proximity operator for constant zero.
 *
 * NOTE In matlab tfocs this functionality is implemented in prox_0.m.
 * @see [[https://github.com/cvxr/TFOCS/blob/master/prox_0.m]]
 */
class ProxZero extends ProxCapableFunction[DenseVector] {

  override def apply(z: DenseVector, t: Double, mode: ProxMode): ProxValue[DenseVector] =
    ProxValue(Some(0.0), Some(z))

  override def apply(x: DenseVector): Double = 0.0
}

/**
 * The proximity operator for the L1 norm, with a scale 'q' applied.
 *   q * ||x||_1
 *
 * @param q A positive scalar factor to multiply by the computed L1 norm. Must be > 0.
 *
 * NOTE In matlab tfocs this functionality is implemented in prox_l1.m.
 * @see [[https://github.com/cvxr/TFOCS/blob/master/prox_l1.m]]
 */
class ProxL1(q: Double) extends ProxCapableFunction[DenseVector] {

  require(q > 0)

  override def apply(z: DenseVector, t: Double, mode: ProxMode): ProxValue[DenseVector] = {
    // NOTE DenseVectors are assumed here (not sparse safe).
    val shrinkage = q * t
    val minimizer = shrinkage match {
      case 0.0 => z
      case _ => new DenseVector(z.values.map(z_i =>
        z_i * (1.0 - math.min(shrinkage / math.abs(z_i), 1.0))))
    }
    val f = if (mode.f) Some(apply(minimizer)) else None
    ProxValue(f, Some(minimizer))
  }

  override def apply(x: DenseVector): Double = q * Vectors.norm(x, 1)
}

/**
 * A projection onto the nonnegative orthant, implemented using a zero/infinity indicator function.
 *
 * NOTE In matlab tfocs this functionality is implemented in proj_Rplus.m.
 * @see [[https://github.com/cvxr/TFOCS/blob/master/proj_Rplus.m]]
 */
class ProjRPlus extends ProxCapableFunction[DenseVector] {

  override def apply(z: DenseVector, t: Double, mode: ProxMode): ProxValue[DenseVector] = {

    val minimizer = if (mode.minimizer) {
      // NOTE DenseVectors are assumed here (not sparse safe).
      Some(new DenseVector(z.values.map(math.max(_, 0.0))))
    } else {
      None
    }

    ProxValue(Some(0.0), minimizer)
  }

  override def apply(x: DenseVector): Double =
    if (x.values.min < 0.0) Double.PositiveInfinity else 0.0
}

/**
 * A projection onto a simple box defined by upper and lower limits on each vector element,
 * implemented using a zero/infinity indicator function.
 *
 * @param l A vector describing the box's lower bound. If x is within the box then x_i >= l_i for
 *        all i.
 * @param u A vector describing the box's upper bound. If x is within the box then x_i <= u_i for
 *        all i.
 *
 * NOTE In matlab tfocs this functionality is implemented in proj_box.m.
 * @see [[https://github.com/cvxr/TFOCS/blob/master/proj_box.m]]
 */
class ProjBox(l: DenseVector, u: DenseVector) extends ProxCapableFunction[DenseVector] {

  override def apply(z: DenseVector, t: Double, mode: ProxMode): ProxValue[DenseVector] = {

    val minimizer = if (mode.minimizer) {
      // NOTE DenseVectors are assumed here (not sparse safe).
      val ret = new Array[Double](z.size)
      var i = 0
      while (i < ret.size) {
        // Bound each element using the lower and upper limit for that element.
        ret(i) = math.min(u(i), math.max(l(i), z(i)))
        i += 1
      }
      Some(new DenseVector(ret))
    } else {
      None
    }

    ProxValue(Some(0.0), minimizer)
  }

  override def apply(x: DenseVector): Double = {
    // NOTE DenseVectors are assumed here (not sparse safe).
    var ret = 0.0
    var i = 0
    while (i < x.size) {
      // If an element is outside of that element's bounds, return infinity.
      if (x(i) > u(i) || x(i) < l(i)) {
        ret = Double.PositiveInfinity
      }
      i += 1
    }
    ret
  }
}
