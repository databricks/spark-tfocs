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

import org.apache.spark.mllib.linalg.{ DenseVector, Vectors }
import org.apache.spark.mllib.optimization.tfocs.fs.dvector.double._
import org.apache.spark.mllib.optimization.tfocs.fs.vector.double._
import org.apache.spark.mllib.optimization.tfocs.fs.vector.dvector._
import org.apache.spark.mllib.optimization.tfocs.VectorSpace._
import org.apache.spark.mllib.optimization.tfocs.vs.dvector._
import org.apache.spark.mllib.optimization.tfocs.vs.vector._

/** Helper to solve lasso regression problems. */
object SolverL1RLS {

  /**
   * Solves the l1 regularized least squares problem 0.5 * ||A * x' - b||_2^2 + lambda * ||x||_1
   * using the Spark TFOCS optimizer. This problem is sometimes referred to as lasso linear
   * regression.
   *
   * @param A The design matrix, represented as a DMatrix.
   * @param b The observed values, represented as a DVector.
   * @param lambda The regularization term.
   * @param x0 Initial value of 'x'. A default value will be used if not provided.
   *
   * @return A tuple containing two elements. The first element is a vector containing the optimized
   *         'x' value. The second element contains the objective function history.
   *
   * @see [[org.apache.spark.mllib.optimization.tfocs.examples.TestLASSO]]
   * for example usage of this function.
   *
   * NOTE The distributed matrix 'A', represented as a DMatrix, and the distributed vector 'b',
   * represented as a DVector, must be consistently partitioned. The 'A' matrix must contain the
   * same number of rows in each partition as the 'b' vector has numeric values in the corresponding
   * partition.
   *
   * @see [[org.apache.spark.mllib.optimization.tfocs.VectorSpace]] for more information about the
   * storage formats used by DVector and DMatrix.
   *
   * NOTE In matlab tfocs this functionality is implemented in solver_L1RLS.m.
   * @see [[https://github.com/cvxr/TFOCS/blob/master/solver_L1RLS.m]]
   */
  def run(A: DMatrix,
    b: DVector,
    lambda: Double,
    x0: Option[DenseVector] = None): (DenseVector, Array[Double]) = {
    val (x, TFOCS.OptimizationData(lossHistory, _, _)) =
      TFOCS.optimize(new SmoothQuad(b),
        new LinopMatrix(A),
        new ProxL1(lambda),
        x0.getOrElse(Vectors.zeros(A.first().size).toDense))
    (x, lossHistory)
  }
}
