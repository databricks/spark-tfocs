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

package org.apache.spark.mllib.optimization.tfocs.fs.generic.double

import org.apache.spark.mllib.optimization.tfocs.{ Mode, ProxCapableFunction, ProxMode, ProxValue, SmoothFunction, Value, VectorSpace }
import org.apache.spark.mllib.optimization.tfocs.VectorSpace._
import org.apache.spark.storage.StorageLevel

/**
 * A smooth objective function created by applying smoothing to a prox capable function at a prox
 * center. This smoothing function is the basis of the smooth conic dual solver in TFOCS_SCD.scala.
 *
 * @param objectiveF The prox capable function.
 * @param mu The smoothing parameter.
 * @param x0 The prox center.
 * @tparam X Type representing a vector on which the function operates.
 *
 * TODO Make the implementation generic to support arbitrary vector spaces, using the VectorSpace
 * abstraction.
 *
 * NOTE In matlab tfocs this functionality is implemented in tfocs_SCD.m.
 * @see [[https://github.com/cvxr/TFOCS/blob/master/tfocs_SCD.m]]
 */
class SmoothDual[X](objectiveF: ProxCapableFunction[X], mu: Double, x0: X)(
    implicit vs: VectorSpace[X]) extends SmoothFunction[X] {

  vs.cache(x0)

  override def apply(ATz: X, mode: Mode): Value[X] = {

    val offsetCenter = vs.combine(mu, ATz, 1.0, x0)
    val ProxValue(proxF, Some(proxMinimizer)) = objectiveF(offsetCenter, mu, ProxMode(mode.f, true))

    // Cache proxMinimizer when it will be required more than once.
    if (mode.f) vs.cache(proxMinimizer)

    val f = if (mode.f) {
      // TODO This might be optimized as a single spark job.
      val diff = vs.combine(1.0, x0, -1.0, proxMinimizer)
      Some(vs.dot(ATz, proxMinimizer) - proxF.get - (0.5 / mu) * vs.dot(diff, diff))
    } else {
      None
    }

    val g = if (mode.g) {
      Some(vs.combine(-1.0, proxMinimizer, 0.0, proxMinimizer))
    } else {
      None
    }

    Value(f, g)
  }
}
