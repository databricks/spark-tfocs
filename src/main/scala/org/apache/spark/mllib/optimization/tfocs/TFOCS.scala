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
   * Metadata describing an execution of TFOCS.optimize.
   *
   * @param lossHistory The history of the objective function values recorded during optimization.
   * @param dual (optional) The value of the dual vector corresponding to the optimal primal vector
   *        returned by the optimizer. Nonempty when isDual = true is specified in TFOCS.optimize.
   * @param L The optimizer's final Lipschitz estimate.
   */
  case class OptimizationData[C](lossHistory: Array[Double], dual: Option[C], L: Double)

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
   * @param convergenceTol The tolerance to use for the convergence test. The exact meaning of
   *        convergenceTol depends on the convergence test type:
   *        isDual == false (default) The convergence test is applied to the primal ('x') rather
   *          than dual values. The test is based on measuring the change in 'x' over successive
   *          iterations. If the magnitude of 'x' is above 1, the algorithm converges when the
   *          magnitude of the relative difference between successive 'x' vectors drops below
   *          convergenceTol. But if the magnitude of 'x' is 1 or below, the magnitude of the
   *          absolute difference between successive 'x' vectors is tested for convergence instead
   *          of the magnitude of the relative difference.
   *        isDual == true The convergence test is applied to dual rather than primal values. The
   *          algorithm converges when the magnitude of the relative difference between successive
   *          dual vectors drops below convergenceTol.
   * @param L0 The initial Lipschitz estimate. This initial value will be adjusted via backtracking
   *        line search. The default L0 value generally provides satisfactory performance.
   * @param isDual Setting this to true indicates that a dual problem is being optimized and the
   *        following actions are taken:
   *        - Negate the objective function to perform concave maximization.
   *        - Record the dual values, returning the optimal dual value with OptimizationData.
   *        - Check convergence using the dual rather than primal values.
   * @param dualTolCheckInterval The iteration interval between convergence tests when optimizing
   *        a dual (isDual == true). Used to throttle potentially slow convergence tests.
   * @param rows The VectorSpace used for computation on row vectors. The 'x' vectors to be
   *        optimized belong to this row VectorSpace.
   * @param cols The VectorSpace used for computation on column vectors.
   * @tparam R Type representing a row vector.
   * @tparam C Type representing a column vector.
   *
   * @return A tuple containing two elements. The first element is a vector containing the optimized
   *         'x' values. The second element is an OptimizationData object containing metadata about
   *         the optimization task, including the objective function history.
   *
   * NOTE In matlab tfocs this functionality is implemented in tfocs.m, tfocs_initialize.m,
   *      tfocs_AT.m, tfocs_backtrack.m, tfocs_iterate.m, and tfocs_cleanup.m.
   * @see [[https://github.com/cvxr/TFOCS/blob/master/tfocs.m]]
   * @see [[https://github.com/cvxr/TFOCS/blob/master/private/tfocs_initialize.m]]
   * @see [[https://github.com/cvxr/TFOCS/blob/master/tfocs_AT.m]]
   * @see [[https://github.com/cvxr/TFOCS/blob/master/private/tfocs_backtrack.m]]
   * @see [[https://github.com/cvxr/TFOCS/blob/master/private/tfocs_iterate.m]]
   * @see [[https://github.com/cvxr/TFOCS/blob/master/private/tfocs_cleanup.m]]
   */
  def optimize[R, C](
    f: SmoothFunction[C],
    A: LinearOperator[R, C],
    h: ProxCapableFunction[R],
    x0: R,
    numIterations: Int = 200,
    convergenceTol: Double = 1e-8,
    L0: Double = 1.0,
    isDual: Boolean = false,
    dualTolCheckInterval: Int = 10)(
      implicit rows: VectorSpace[R],
      cols: VectorSpace[C]): (R, OptimizationData[C]) = {

    val maxmin = if (isDual) -1 else 1

    val Lexact = Double.PositiveInfinity
    val beta = 0.5
    val alpha = 0.9

    var x = x0
    var z = x
    rows.cache(x)
    var a_x = A(x)
    var a_z = a_x
    cols.cache(a_x)
    var dual: Option[C] = None
    var theta = Double.PositiveInfinity
    val lossHistory = new ArrayBuffer[Double](numIterations)

    var L = L0

    var backtrack_simple = true
    val backtrack_tol = 1e-10

    var cntrAy = 0
    var cntrAx = 0
    val cntrReset = 50

    var cntrTol = 0

    var hasConverged = false
    for (nIter <- 1 to numIterations if !hasConverged) {

      val (x_old, z_old) = (x, z)
      val (a_x_old, a_z_old) = (a_x, a_z)
      val oldDual = dual
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
        val a_y = if (cntrAy >= cntrReset) {
          cntrAy = 0
          A(y)
        } else {
          cntrAy = cntrAy + 1
          cols.combine(1.0 - theta, a_x_old, theta, a_z_old)
        }
        if (!backtrack_simple) cols.cache(a_y)

        val Value(Some(f_y_), Some(g_Ay_)) = f(a_y, Mode(true, true))

        f_y = f_y_
        val g_Ay = cols.combine(maxmin, g_Ay_, 0.0, g_Ay_)
        dual = Some(g_Ay)
        if (!backtrack_simple) cols.cache(g_Ay)
        g_y = A.t(g_Ay)
        rows.cache(g_y)
        val step = 1.0 / (theta * L)
        z = h(rows.combine(1.0, z_old, -step, g_y), step, ProxMode(false, true)).minimizer.get
        rows.cache(z)
        a_z = A(z)
        cols.cache(a_z)

        x = rows.combine(1.0 - theta, x_old, theta, z)
        rows.cache(x)
        a_x = if (cntrAx >= cntrReset) {
          cntrAx = 0
          A(x)
        } else {
          cntrAx = cntrAx + 1
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
              val Value(_, Some(g_Ax_)) = f(a_x, Mode(false, true))
              val g_Ax = cols.combine(maxmin, g_Ax_, 0.0, g_Ax_)
              dual = Some(g_Ax)
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
      // use f_y. The prox capable function value is included in this loss computation.
      lossHistory.append(f_x match {
        case Some(f) => maxmin * (f + h(x))
        case _ => maxmin * (f_y + h(y))
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
      hasConverged = if (!isDual) {

        // Check convergence tolerance on the primal vector x.
        val norm_x = math.sqrt(rows.dot(x, x))
        val dx = rows.combine(1.0, x, -1.0, x_old)
        rows.cache(dx)
        math.sqrt(rows.dot(dx, dx)) match {
          case 0.0 => nIter > 1
          case norm_dx => norm_dx < convergenceTol * math.max(norm_x, 1)
        }
      } else {

        // Check convergence tolerance on the dual vector. Because this check requires multiple
        // spark jobs when the dual is a distributed vector, it is only performed once every
        // dualTolCheckInterval iterations.
        if (cntrTol + 1 >= dualTolCheckInterval) {
          cntrTol = 0

          var d_dual = Double.PositiveInfinity
          if (dual.isDefined && oldDual.isDefined) {
            cols.cache(dual.get)
            cols.cache(oldDual.get)
            val normCur = math.sqrt(cols.dot(dual.get, dual.get))
            val normOld = math.sqrt(cols.dot(oldDual.get, oldDual.get))
            if (normCur > 2e-15 && normOld > 2e-15) {
              val dualDiff = cols.combine(1.0, oldDual.get, -1.0, dual.get)
              d_dual = math.sqrt(cols.dot(dualDiff, dualDiff)) / normCur
            }
          }
          d_dual < convergenceTol && nIter > 2
        } else {

          cntrTol = cntrTol + 1
          false
        }
      }

      // Abort iteration if the computed loss function value is invalid.
      if (f_y.isNaN || f_y.isInfinity) {
        logWarning("Unable to compute loss function.")
        hasConverged = true
      }
    }

    logInfo("TFOCS.optimize finished. Last 10 losses %s".format(
      lossHistory.takeRight(10).mkString(", ")))

    if (isDual) {
      // Always set the final dual using x, for consistency with the returned primal.
      val Value(_, Some(g_Ax_)) = f(a_x, Mode(false, true))
      val g_Ax = cols.combine(maxmin, g_Ax_, 0.0, g_Ax_)
      dual = Some(g_Ax)
    }

    (x, OptimizationData(lossHistory.toArray, if (isDual) dual else None, L))
  }
}
