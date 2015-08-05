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
import org.apache.spark.mllib.optimization.tfocs.{ Mode, ProxCapableFunction, ProxMode, ProxValue, SmoothFunction, Value }
import org.apache.spark.mllib.optimization.tfocs.VectorSpace._
import org.apache.spark.storage.StorageLevel

/**
 * A smooth objective function created by applying smoothing to a prox capable function at a prox
 * center. This smoothing function is the basis of the smooth conic dual solver in TFOCS_SCD.scala.
 *
 * @param objectiveF The prox capable function.
 * @param mu The smoothing parameter.
 * @param x0 The prox center.
 *
 * TODO Make the implementation generic to support arbitrary vector spaces, using the VectorSpace
 * abstraction.
 *
 * NOTE In matlab tfocs this functionality is implemented in tfocs_SCD.m.
 * @see [[https://github.com/cvxr/TFOCS/blob/master/tfocs_SCD.m]]
 */
class SmoothDual(objectiveF: ProxCapableFunction[DVector], mu: Double, x0: DVector)
    extends SmoothFunction[DVector] with Serializable {

  if (x0.getStorageLevel == StorageLevel.NONE) {
    x0.cache()
  }

  override def apply(ATz: DVector, mode: Mode): Value[DVector] = {

    val offsetCenter = axpy(mu, ATz, x0)
    val ProxValue(proxF, Some(proxMinimizer)) = objectiveF(offsetCenter, mu, ProxMode(mode.f, true))

    // Cache proxMinimizer when it will be required more than once.
    if (mode.f) proxMinimizer.cache()

    val f = if (mode.f) {
      // TODO This might be optimized as a single spark job.
      Some(ATz.dot(proxMinimizer) - proxF.get - (0.5 / mu) * x0.sqdist(proxMinimizer))
    } else {
      None
    }

    val g = if (mode.g) {
      Some(proxMinimizer.mapElements(proxMinimizer_i => -proxMinimizer_i))
    } else {
      None
    }

    Value(f, g)
  }
}
