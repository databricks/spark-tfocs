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

import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.optimization.tfocs.VectorSpace._
import org.apache.spark.mllib.optimization.tfocs.fs.generic.double._
import org.apache.spark.mllib.optimization.tfocs.fs.dvectordouble.double._
import org.apache.spark.mllib.optimization.tfocs.vs.vector._
import org.apache.spark.mllib.optimization.tfocs.vs.dvectordouble._

object TFOCS_SCD {

  /**
   * The smoothed conic dual form of TFOCS, for problems with non trivial linear operators.
   *
   * Solves a conic problem using the smoothed conic dual approach, in the following conic form:
   *   minimize objectiveF(x) + 0.5 * mu(x-x0).^2
   *   s.t.     affineF(x) \in \cK
   *
   * The user is responsible for constructing the dual proximity function so that the dual can be
   * described using the saddle point problem:
   *   maximize_z inf_x [objectiveF(x) + 0.5 * mu(x - x0).^2 - <affineF(x), z>] - dualproxF(z)
   *
   * @param objectiveF Prox capable objective function.
   * @param affineF Linear component of the objective function.
   * @param dualProxF Proximity function for the dual problem.
   * @param mu Smoothing parameter.
   * @param x0 Starting x value.
   * @param z0 Starting dual (z) value.
   * @param numContinuations The maximum number of continuations to use during optimization.
   * @param tol The convergence tolerance threshold.
   * @param initialTol The convergence tolerance threshold for the first continuation iteration.
   * @param dualTolCheckInterval The iteration interval between convergence tests in TFOCS.optimize.
   *        Used to throttle potentially slow convergence tests.
   * @param cols The VectorSpace used for computation on column vectors.
   *
   * @return A tuple containing two elements. The first element is a vector containing the optimized
   *         'x' value. The second element contains the objective function history.
   *
   * NOTE In matlab tfocs this functionality is implemented in tfocs_SCD.m and continuation.m.
   * @see [[https://github.com/cvxr/TFOCS/blob/master/tfocs_SCD.m]]
   * @see [[https://github.com/cvxr/TFOCS/blob/master/continuation.m]]
   *
   * TODO Support general vector spaces via type parameters.
   */
  def optimize(
    objectiveF: ProxCapableFunction[DVector],
    affineF: LinearOperator[(DVector, Double), DenseVector],
    dualProxF: ProxCapableFunction[DenseVector],
    mu: Double,
    x0: DVector,
    z0: DenseVector,
    numContinuations: Int,
    tol: Double,
    initialTol: Double,
    dualTolCheckInterval: Int)(
      implicit cols: VectorSpace[DVector]): (DVector, Array[Double]) = {

    var x0Iter = x0
    var z0Iter = z0
    var x = x0
    var xOld = x0
    var L = 1.0
    var hist = new Array[Double](0)

    // Find betaTol, the factor by which to decrease the convergence tolerance on each iteration.
    val betaTol = math.exp(math.log(initialTol / tol) / (numContinuations - 1))
    // Find the initial convergence tolerance.
    var iterTol = tol * math.pow(betaTol, numContinuations)

    var hasConverged = false
    for (nIter <- 1 to numContinuations if !hasConverged) {

      // Run the convex optimizer until the iterTol tolerance is reached.
      iterTol = iterTol / betaTol
      val smoothFunction = new SmoothCombine(new SmoothDual(objectiveF, 1 / mu, x0Iter))
      val (z, optimizationData) = TFOCS.optimize(smoothFunction,
        affineF.t,
        dualProxF,
        z0Iter,
        TFOCSMaxIterations,
        iterTol,
        L,
        true,
        dualTolCheckInterval)

      // Update the optimization loop parameters.
      x = optimizationData.dual.get._1
      cols.cache(x)
      hist ++= optimizationData.lossHistory
      L = optimizationData.L

      // Update the prox center, applying acceleration to x.
      x0Iter = cols.combine(1.0 + (nIter - 1.0) / (nIter + 2.0), x,
        (1.0 - nIter) / (nIter + 2.0), xOld)
      z0Iter = z

      // Check for convergence.
      val dx = cols.combine(1, x, -1, xOld)
      val n1 = math.sqrt(cols.dot(dx, dx))
      val n2 = math.sqrt(cols.dot(xOld, xOld))
      hasConverged = n1 / n2 <= tol

      xOld = x
    }

    (x, hist)
  }

  private val TFOCSMaxIterations = 2000
}
