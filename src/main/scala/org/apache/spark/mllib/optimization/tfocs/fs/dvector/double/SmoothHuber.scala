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

import org.apache.spark.mllib.optimization.tfocs.DVectorFunctions._
import org.apache.spark.mllib.optimization.tfocs.{ Mode, SmoothFunction, Value }
import org.apache.spark.mllib.optimization.tfocs.VectorSpace._
import org.apache.spark.storage.StorageLevel

/**
 * The Huber loss function, corresponding to the adjusted l2 loss when the magnitude is <= tau
 * and to the adjusted l1 loss when the magnitude is > tau:
 *   sum_i  0.5 * ((x_i - x0_i) ^ 2) / tau   if |x_i - x0_i| <= tau
 *          |x_i - x0_i| - tau / 2           if |x_i - x0_i| >  tau
 *
 * @param x0 The vector against which the loss should be computed.
 * @param tau The Huber loss parameter.
 *
 * NOTE In matlab tfocs this functionality is implemented in smooth_huber.m.
 * @see [[https://github.com/cvxr/TFOCS/blob/master/smooth_huber.m]]
 */
class SmoothHuber(x0: DVector, tau: Double) extends SmoothFunction[DVector] {

  if (x0.getStorageLevel == StorageLevel.NONE) {
    x0.cache()
  }

  override def apply(x: DVector, mode: Mode): Value[DVector] = {

    val diff = x.diff(x0)
    val tau = this.tau

    // If both f and g are requested then diff will be read twice, so cache it.
    if (mode.f && mode.g) diff.cache()

    val f = if (mode.f) {
      // TODO If f is required but not g, then performance might be improved by reimplementing as
      // a single aggregate using 'x' and 'x0' without an intermediate 'diff' DVector, which breaks
      // per-element pipelining.
      Some(diff.aggregateElements(0.0)(
        seqOp = (sum, diff_i) => {
          // Find the Huber loss, corresponding to the adjusted l2 loss when the magnitude is <= tau
          // and to the adjusted l1 loss when the magnitude is > tau.
          val huberValue = if (math.abs(diff_i) <= tau) {
            0.5 * diff_i * diff_i / tau
          } else {
            math.abs(diff_i) - tau / 2.0
          }
          sum + huberValue
        },
        combOp = _ + _))
    } else {
      None
    }

    val g = if (mode.g) {
      // Compute the Huber loss gradient elementwise.
      Some(diff.mapElements(diff_i => diff_i / math.max(math.abs(diff_i), tau)))
    } else {
      None
    }

    Value(f, g)
  }
}
