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
