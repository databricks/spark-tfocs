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
 * A trait for smooth functions, with support for evaluating the function and computing its
 * gradient.
 *
 * @tparam X Type representing a vector on which to evaluate the function.
 */
trait SmoothFunction[X] {
  /**
   * Evaluates this function at x and returns the function value and gradient, depending on the mode
   * specified.
   *
   * @param x The vector on which to evaluate the function.
   * @param mode The computation mode. If mode.f is true, the function value is returned. If mode.g
   *        is true, the function gradient is returned.
   * @return A Value containing the function value and/or gradient, depending on the 'mode'
   *         parameter. The function value is contained in value.f, while the gradient is contained
   *         in value.g.
   */
  def apply(x: X, mode: Mode): Value[X]

  /** Evaluates the function on vector x. */
  def apply(x: X): Double = apply(x, Mode(f = true, g = false)).f.get
}
