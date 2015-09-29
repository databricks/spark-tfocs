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

/**
 * A prox capable function trait with support for efficient proximity minimization, as expressed
 * by the proximity operator:
 *   x = prox_h(z, t) = argmin_x(h(x) + 0.5 * ||x - z||_2^2 / t)
 *
 * The minimizing x value and/or the function value h(x) may be computed, depending on the
 * specified ProxMode.
 *
 * @tparam X A type representing a vector on which to evaluate the function.
 *
 * @see [[http://cvxr.com/tfocs/doc]] for more information on prox capable functions.
 */
trait ProxCapableFunction[X] {

  /**
   * Evaluate the proximity operator prox_h at z with parameter t, returning x and/or h(x)
   * depending on the mode specified.
   *
   * @param z The vector on which to evaluate the proximity operator.
   * @param t The proximity parameter.
   * @param mode The computation mode. If mode.f is true, h(x) is returned. If mode.minimizer is
   *        true, x is returned.
   *
   * @return A ProxValue containing x, the vector minimizing the proximity function prox_h, and/or
   *         h(x), the function value at x. The exact set of values computed and returned depends on
   *         the attributes of the supplied 'mode' parameter. The returned Value contains h(x) in
   *         its 'f' attribute, while x is contained in the 'minimizer' attribute.
   */
  def apply(z: X, t: Double, mode: ProxMode): ProxValue[X]

  /**
   * Evaluate the function h(x) at x. Does not perform proximity minimization.
   *
   * @param x The vector on which to evaluate the function h.
   *
   * @return The value of h(x).
   */
  def apply(x: X): Double
}
