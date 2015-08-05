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

package org.apache.spark.mllib.optimization.tfocs.fs.dvectordouble.double

import org.apache.spark.mllib.optimization.tfocs.{ Mode, SmoothFunction, Value }
import org.apache.spark.mllib.optimization.tfocs.VectorSpace._

/**
 * A smooth function operating on (DVector, Double) pairs. Such pairs arise when two separate
 * linear operators are applied to an input value. The value returned by SmoothCombine is equal to
 * the value of the objective function evaluated on the first (DVector) element of the pair summed
 * with the second (Double) element of the pair.
 *
 * NOTE In matlab tfocs this functionality is implemented in smooth_stack.m and smooth_linear.m.
 * @see [[https://github.com/cvxr/TFOCS/blob/master/private/smooth_stack.m]]
 * @see [[https://github.com/cvxr/TFOCS/blob/master/smooth_linear.m]]
 */
class SmoothCombine(objectiveF: SmoothFunction[DVector]) extends SmoothFunction[(DVector, Double)] {

  override def apply(x: (DVector, Double), mode: Mode): Value[(DVector, Double)] = {
    val (xVector, xAncillaryScalar) = x
    val Value(f, g) = objectiveF(xVector, mode)
    Value(f.map(_ + xAncillaryScalar), g.map((_, 1.0)))
  }
}
