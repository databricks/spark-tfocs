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

import scala.collection.mutable.ArrayBuffer

import org.apache.spark.Logging

object TFOCS extends Logging {

  /**
   * Optimize an objective function using accelerated proximal gradient descent.
   * The implementation is based on TFOCS [[http://cvxr.com/tfocs]], described in Becker, Candes,
   * and Grant 2010. A limited but useful subset of the TFOCS feature set is implemented, including
   * support for composite loss functions, the Auslender and Teboulle acceleration method,
   * backtracking Lipschitz estimation, and automatic restart using the gradient test.
   *
   * The composite ojective function is formed by combining multiple parameters.
   * The full objective function is: f(A * x) + h(x)
   *
   * @param f The smooth portion of the objective function.
   * @param A The linear portion of the objective function.
   * @param h The nonsmooth (prox-capable) portion of the objective function.
   * @param x0 Starting value of the vector to optimize.
   * @param numIterations The maximum number of iterations to run the optimization algorithm.
   * @param convergenceTol The tolerance to use for the convergence test. The convergence test is
   *        based on measuring the change in 'x' over successive iterations. If the magnitude of 'x'
   *        is above 1, the algorithm converges when the magnitude of the relative difference
   *        between successive 'x' vectors drops below convergenceTol. But if the magnitude of 'x'
   *        is 1 or below, the magnitude of the absolute difference between successive 'x' vectors
   *        is tested for convergence instead of the magnitude of the relative difference.
   * @param rows The VectorSpace used for computation on row vectors. The 'x' vectors to be
   *        optimized belong to this row VectorSpace.
   * @param cols The VectorSpace used for computation on column vectors.
   * @tparam R Type representing a row vector.
   * @tparam C Type representing a column vector.
   *
   * @return A tuple containing two elements. The first element is a row vector containing the
   *         minimizing 'x' values. The second element contains the objective function history.
   */
  def optimize[R, C](
    f: SmoothFunction[C],
    A: LinearFunction[R, C],
    h: ProxCapableFunction[R],
    x0: R,
    numIterations: Int = 200,
    convergenceTol: Double = 1e-8)(
      implicit rows: VectorSpace[R],
      cols: VectorSpace[C]): (R, Array[Double]) = {

    val L0 = 1.0
    val Lexact = Double.PositiveInfinity
    val beta = 0.5
    val alpha = 0.9

    var x = x0
    var z = x
    rows.cache(x)
    var a_x = A(x)
    var a_z = a_x
    cols.cache(a_x)
    var theta = Double.PositiveInfinity
    val lossHistory = new ArrayBuffer[Double](numIterations)

    var L = L0

    var backtrack_simple = true
    val backtrack_tol = 1e-10

    var cntr_Ay = 0
    var cntr_Ax = 0
    val cntr_reset = 50

    var hasConverged = false
    for (nIter <- 1 to numIterations if !hasConverged) {

      val (x_old, z_old) = (x, z)
      val (a_x_old, a_z_old) = (a_x, a_z)
      val L_old = L
      L = L * alpha
      val theta_old = theta

      var y = x0
      var f_x: Option[Double] = None
      var f_y = 0.0
      var g_y = x0

      var isBacktracking = true
      while (isBacktracking) {

        f_x = None

        // Auslender and Teboulle's accelerated method.

        theta = 2.0 / (1.0 + math.sqrt(1.0 + 4.0 * (L / L_old) / (theta_old * theta_old)))

        y = rows.combine(1.0 - theta, x_old, theta, z_old)
        val a_y = if (cntr_Ay >= cntr_reset) {
          cntr_Ay = 0
          A(y)
        } else {
          cntr_Ay = cntr_Ay + 1
          cols.combine(1.0 - theta, a_x_old, theta, a_z_old)
        }
        if (!backtrack_simple) cols.cache(a_y)

        val Value(Some(f_y_), Some(g_Ay)) = f(a_y, Mode(true, true))

        f_y = f_y_
        if (!backtrack_simple) cols.cache(g_Ay)
        g_y = A.t(g_Ay)
        rows.cache(g_y)
        val step = 1.0 / (theta * L)
        z = h(rows.combine(1.0, z_old, -step, g_y), step, Mode(false, true)).g.get
        rows.cache(z)
        a_z = A(z)
        cols.cache(a_z)

        x = rows.combine(1.0 - theta, x_old, theta, z)
        rows.cache(x)
        a_x = if (cntr_Ax >= cntr_reset) {
          cntr_Ax = 0
          A(x)
        } else {
          cntr_Ax = cntr_Ax + 1
          cols.combine(1.0 - theta, a_x_old, theta, a_z)
        }
        cols.cache(a_x)

        // If a non divergence criterion is violated, adjust the Lipschitz estimate and re-run the
        // inner (backtracking) loop.
        isBacktracking = false
        if (beta < 1.0) {

          val xy = rows.combine(1.0, x, -1.0, y)
          rows.cache(xy)
          val xy_sq = rows.dot(xy, xy)
          if (xy_sq != 0.0) {

            // Compute localL using one of two non divergence criteria. The backtrack_simple
            // criterion is more accurate but prone to numerical instability.
            var localL = if (backtrack_simple) {
              f_x = Some(f(a_x))
              backtrack_simple =
                (math.abs(f_y - f_x.get) >=
                  backtrack_tol * math.max(math.abs(f_x.get), math.abs(f_y)))
              val q_x = f_y + rows.dot(xy, g_y) + 0.5 * L * xy_sq
              L + 2.0 * math.max(f_x.get - q_x, 0.0) / xy_sq
            } else {
              val Value(_, Some(g_Ax)) = f(a_x, Mode(false, true))
              2.0 * cols.dot(cols.combine(1.0, a_x, -1.0, a_y),
                cols.combine(1.0, g_Ax, -1.0, g_Ay)) / xy_sq
            }

            // If L violates a non divergence criterion above, adjust L and re-run the backtracking
            // loop. Otherwise do not backtrack.
            if (localL > L && L < Lexact) {

              isBacktracking = true
              if (!localL.isInfinity) {
                L = math.min(Lexact, localL)
              } else if (localL.isInfinity) {
                localL = L
              }
              L = math.min(Lexact, math.max(localL, L / beta))
            }
          }
        }
      }

      // Track the loss history using the smooth function value at x (f_x) if available. Otherwise
      // use f_y. The prox capable function is included in this loss computation.
      lossHistory.append(f_x match {
        case Some(f) => f + h(x)
        case _ => f_y + h(y)
      })

      // Restart acceleration if indicated by the gradient test from O'Donoghue and Candes 2013.
      // NOTE TFOCS uses <g_Ay, a_x - a_x_old> here, but we do it this way to avoid a spark job.
      if (rows.dot(g_y, rows.combine(1.0, x, -1.0, x_old)) > 0.0) {
        z = x
        a_z = a_x
        theta = Double.PositiveInfinity
        backtrack_simple = true
      }

      // Check for convergence.
      val norm_x = math.sqrt(rows.dot(x, x))
      val dx = rows.combine(1.0, x, -1.0, x_old)
      rows.cache(dx)
      hasConverged = math.sqrt(rows.dot(dx, dx)) match {
        case 0.0 => nIter > 1
        case norm_dx => norm_dx < convergenceTol * math.max(norm_x, 1)
      }

      // Abort iteration if the computed loss function value is invalid.
      if (f_y.isNaN || f_y.isInfinity) {
        logWarning("Unable to compute loss function.")
        hasConverged = true
      }
    }

    logInfo("TFOCS.minimize finished. Last 10 losses %s".format(
      lossHistory.takeRight(10).mkString(", ")))

    (x, lossHistory.toArray)
  }
}
