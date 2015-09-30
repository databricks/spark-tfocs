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

import org.apache.spark.mllib.linalg.{ BLAS, DenseVector, Vectors }
import org.apache.spark.mllib.optimization.tfocs.DVectorFunctions._
import org.apache.spark.mllib.optimization.tfocs.VectorSpace._
import org.apache.spark.mllib.optimization.tfocs.fs.dvector.double._
import org.apache.spark.mllib.optimization.tfocs.fs.dvectordouble.vector._
import org.apache.spark.mllib.optimization.tfocs.fs.vector.double._
import org.apache.spark.mllib.optimization.tfocs.vs.dvector._

object SolverSLP {

  /**
   * Solve the smoothed standard form linear program:
   *   minimize c' * x + 0.5 * mu * ||x - x0||_2^2
   *   s.t.     A' * x == b' and x >= 0
   *
   * @param c Objective function coefficient vector. Represented as a distributed vector, DVector.
   *        See note below.
   * @param A Constraint coefficient matrix. 'A' is an m by n matrix, where m is the length of x and
   *        n is the length of b. That is, the transpose of A may be multiplied by x. 'A' is
   *        represented as a distributed matrix, DMatrix. See note below.
   * @param b Constraint coefficient vector.
   * @param mu Smoothing parameter.
   * @param x0 Starting x value. Represented as a distributed vector, DVector. See note below. A
   *        default value will be used if not provided.
   * @param z0 Starting dual (z) value. A default value will be used if not provided.
   * @param numContinuations The maximum number of continuations to use in the TFOCS_SCD
   *        implementation.
   * @param tol The convergence tolerance threshold.
   * @param initialTol The convergence tolerance threshold for the first continuation iteration.
   * @param dualTolCheckInterval The iteration interval between convergence tests in TFOCS.optimize.
   *        Used to throttle potentially slow convergence tests.
   *
   * @return A tuple containing two elements. The first element is a vector containing the optimized
   *         'x' value. The second element contains the internal objective function history.
   *
   * @see [[org.apache.spark.mllib.optimization.tfocs.examples.TestLinearProgram]]
   * @see [[org.apache.spark.mllib.optimization.tfocs.examples.TestMPSLinearProgram]]
   * for example usage of this function.
   *
   * NOTE The distributed matrix, represented as a DMatrix, and the distributed vectors, represented
   * as DVectors, must be consistently partitioned. The DVectors must all contain the same number of
   * numeric values in each partition. And the DMatrix must contain the same number of rows in each
   * partition as the DVectors have numeric values.
   *
   * @see [[org.apache.spark.mllib.optimization.tfocs.VectorSpace]] for more information about the
   * storage formats used by DVector and DMatrix.
   *
   * NOTE In matlab tfocs this functionality is implemented in solver_sLP.m and
   * test_LinearProgram.m.
   * @see [[https://github.com/cvxr/TFOCS/blob/master/solver_sLP.m]]
   * @see [[https://github.com/cvxr/TFOCS/blob/master/examples/smallscale/test_LinearProgram.m]]
   */
  def run(
    c: DVector,
    A: DMatrix,
    b: DenseVector,
    mu: Double,
    x0: Option[DVector] = None,
    z0: Option[DenseVector] = None,
    numContinuations: Int = 10,
    tol: Double = 1e-4,
    initialTol: Double = 1e-3,
    dualTolCheckInterval: Int = 10): (DVector, Array[Double]) = {

    val minusB = b.copy
    BLAS.scal(-1.0, minusB)
    TFOCS_SCD.optimize(new ProxShiftRPlus(c),
      new LinopMatrixAdjoint(A, minusB),
      new ProxZero(),
      mu,
      x0.getOrElse(c.mapElements(_ => 0.0)),
      z0.getOrElse(Vectors.zeros(b.size).toDense),
      numContinuations,
      tol,
      initialTol,
      dualTolCheckInterval)
  }
}
