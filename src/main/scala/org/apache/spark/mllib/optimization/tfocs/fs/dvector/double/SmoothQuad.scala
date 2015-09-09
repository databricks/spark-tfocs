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

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.tfocs.DVectorFunctions._
import org.apache.spark.mllib.optimization.tfocs.{ Mode, SmoothFunction, Value }
import org.apache.spark.mllib.optimization.tfocs.VectorSpace._
import org.apache.spark.storage.StorageLevel

/**
 * The squared error function applied to a DVector, with a constant factor of 0.5.
 *
 * @param x0 The base vector against which the squared error difference is computed.
 *
 * NOTE In matlab tfocs this functionality is implemented in smooth_quad.m.
 * @see [[https://github.com/cvxr/TFOCS/blob/master/smooth_quad.m]]
 */
class SmoothQuad(x0: DVector) extends SmoothFunction[DVector] {

  if (x0.getStorageLevel == StorageLevel.NONE) {
    x0.cache()
  }

  override def apply(x: DVector, mode: Mode): Value[DVector] = {

    // Compute the squared error gradient (just the difference between vectors).
    val g = x.diff(x0)

    // If both f and g are requested then g will be read twice, so cache it.
    if (mode.f && mode.g) g.cache()

    val f = if (mode.f) {
      // Compute the squared error.
      // TODO If f is required but not g, then performance might be improved by reimplementing as
      // a single aggregate using 'x' and 'x0' without an intermediate 'g' DVector, which breaks
      // per-element pipelining.
      Some(g.aggregate(0.0)((sum, gPart) => sum + math.pow(Vectors.norm(gPart, 2), 2), _ + _) / 2.0)
    } else {
      None
    }
    Value(f, Some(g))
  }
}
