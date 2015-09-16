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

package org.apache.spark.mllib.optimization.tfocs.fs.dvector.double

import org.apache.spark.mllib.optimization.tfocs.{ ProxCapableFunction, ProxMode, ProxValue }
import org.apache.spark.mllib.optimization.tfocs.DVectorFunctions._
import org.apache.spark.mllib.optimization.tfocs.VectorSpace._
import org.apache.spark.storage.StorageLevel

/**
 * The proximity operator for the function:
 *   rplus(x) + c' * x
 * where rplus(x) is a zero/infinity indicator function for the nonnegative orthant.
 *
 * @param c The shift vector.
 *
 * NOTE In matlab tfocs this functionality is implemented in prox_shift.m and proj_Rplus.m.
 * @see [[https://github.com/cvxr/TFOCS/blob/master/prox_shift.m]]
 * @see [[https://github.com/cvxr/TFOCS/blob/master/proj_Rplus.m]]
 */
class ProxShiftRPlus(c: DVector) extends ProxCapableFunction[DVector] {

  if (c.getStorageLevel == StorageLevel.NONE) {
    c.cache()
  }

  override def apply(x: DVector, t: Double, mode: ProxMode): ProxValue[DVector] = {
    val minimizer = x.zipElements(c, (x_i, c_i) => math.max(0, x_i - t * c_i))
    // If both f and minimizer are requested, the minimizer will be read twice so cache it.
    if (mode.f && mode.minimizer) minimizer.cache()
    val f = if (mode.f) Some(c.dot(minimizer)) else None
    ProxValue(f, Some(minimizer))
  }

  override def apply(x: DVector): Double = {
    val rPlus = x.aggregateElements(0.0)(
      seqOp = (sum, x_i) => sum + (if (x_i < 0) Double.PositiveInfinity else 0.0),
      combOp = _ + _)
    if (rPlus.isPosInfinity) Double.PositiveInfinity else x.dot(c)
  }
}
