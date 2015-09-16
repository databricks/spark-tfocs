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

package org.apache.spark.mllib.optimization.tfocs.fs.dvectordouble.vector

import org.apache.spark.mllib.linalg.{ BLAS, DenseVector }
import org.apache.spark.mllib.optimization.tfocs.fs.dvector.vector.{ LinopMatrixAdjoint => Delegate }
import org.apache.spark.mllib.optimization.tfocs.fs.vector.dvectordouble.LinopMatrix
import org.apache.spark.mllib.optimization.tfocs.LinearOperator
import org.apache.spark.mllib.optimization.tfocs.VectorSpace._

/**
 * Compute the function A' * x + b, where 'A' is a DMatrix, 'x' is a a DVector, and 'b' is a Vector.
 *
 * NOTE In matlab tfocs this functionality is implemented in linop_stack.m.
 * @see [[https://github.com/cvxr/TFOCS/blob/master/private/linop_stack.m]]
 */
class LinopMatrixAdjoint(private val A: DMatrix, private val b: DenseVector)
    extends LinearOperator[(DVector, Double), DenseVector] {

  private val delegate = new Delegate(A)

  override def apply(x: (DVector, Double)): DenseVector = {
    val ret = delegate.apply(x._1)
    BLAS.axpy(1.0, b, ret)
    ret
  }

  override def t: LinearOperator[DenseVector, (DVector, Double)] = new LinopMatrix(A, b)
}
