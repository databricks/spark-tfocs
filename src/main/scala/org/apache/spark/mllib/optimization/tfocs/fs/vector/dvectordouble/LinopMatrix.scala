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

package org.apache.spark.mllib.optimization.tfocs.fs.vector.dvectordouble

import org.apache.spark.mllib.linalg.BLAS
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.optimization.tfocs.fs.dvectordouble.vector.LinopMatrixAdjoint
import org.apache.spark.mllib.optimization.tfocs.fs.vector.dvector.{ LinopMatrix => Delegate }
import org.apache.spark.mllib.optimization.tfocs.LinearOperator
import org.apache.spark.mllib.optimization.tfocs.VectorSpace._

/**
 * Compute the products A * x' and b * x', where 'A' is a DMatrix, 'x' is a Vector, and 'b' is a
 * Vector. The two products are returned together as a tuple.
 *
 * NOTE In matlab tfocs this functionality is implemented in linop_stack.m.
 * @see [[https://github.com/cvxr/TFOCS/blob/master/private/linop_stack.m]]
 */
class LinopMatrix(private val A: DMatrix, private val b: DenseVector)
    extends LinearOperator[DenseVector, (DVector, Double)] {

  private val delegate = new Delegate(A)

  override def apply(x: DenseVector): (DVector, Double) = (delegate.apply(x), BLAS.dot(b, x))

  override def t: LinearOperator[(DVector, Double), DenseVector] = new LinopMatrixAdjoint(A, b)
}
