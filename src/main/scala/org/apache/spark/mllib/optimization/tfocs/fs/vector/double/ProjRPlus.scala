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
 * A projection onto the nonnegative orthant, implemented using a zero/infinity indicator function:
 *   sum_i  0.0                if x_i >= 0.0
 *          PositiveInfinity   otherwise
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
