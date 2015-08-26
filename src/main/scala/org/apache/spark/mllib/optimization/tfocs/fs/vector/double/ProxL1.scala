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

import org.apache.spark.mllib.linalg.{ DenseVector, Vectors }
import org.apache.spark.mllib.optimization.tfocs.{ ProxCapableFunction, ProxMode, ProxValue }

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
