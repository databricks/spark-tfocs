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

package org.apache.spark.mllib.optimization.tfocs.fs.vector.double

import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.optimization.tfocs.{ ProxCapableFunction, ProxMode, ProxValue }

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
