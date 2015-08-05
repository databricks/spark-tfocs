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

import org.apache.spark.mllib.linalg.{ DenseVector, Vectors, Vector }
import org.apache.spark.mllib.optimization.tfocs.DVectorFunctions._
import org.apache.spark.mllib.optimization.tfocs.VectorSpace._
import org.apache.spark.rdd.RDD

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

  /**
   * Evaluates the function on vector x.
   */
  def apply(x: X): Double = apply(x, Mode(f = true, g = false)).f.get
}

/**
 * The squared error function applied to a DVector, with a constant factor of 0.5.
 *
 * @param x0 The base vector against which the squared error difference is computed.
 */
class SmoothQuadDVector(x0: DVector) extends SmoothFunction[DVector] {

  x0.cache()

  override def apply(x: DVector, mode: Mode): Value[DVector] = {

    // Compute the squared error gradient (just the difference between vectors).
    val g = x.diff(x0)

    // If both f and g are requested then g will be read twice, so cache it.
    if (mode.f && mode.g) g.cache()

    val f = if (mode.f) {
      // Compute the squared error.
      Some(g.treeAggregate(0.0)((sum, y) => sum + Math.pow(Vectors.norm(y, 2), 2), _ + _) / 2.0)
    } else {
      None
    }
    Value(f, Some(g))
  }
}

/**
 * The huber loss function applied to a DVector.
 *
 * @param x0 The vector against which loss should be computed.
 * @param tau The huber loss parameter.
 */
class SmoothHuberDVector(x0: DVector, tau: Double)
    extends SmoothFunction[DVector] with Serializable {

  x0.cache()

  override def apply(x: DVector, mode: Mode): Value[DVector] = {

    val diff = x.diff(x0)

    // If both f and g are requested then diff will be read twice, so cache it.
    if (mode.f && mode.g) diff.cache()

    val f = if (mode.f) {
      Some(diff.aggregateElements(0.0)(
        seqOp = (sum, diff_i) => {
          // Find the huber loss, corresponding to the adjusted l2 loss when the magnitude is <= tau
          // and to the adjusted l1 loss when the magnitude is > tau.
          val huberValue = if (math.abs(diff_i) <= tau) {
            0.5 * diff_i * diff_i / tau
          } else {
            math.abs(diff_i) - tau / 2.0
          }
          sum + huberValue
        },
        combOp = _ + _))
    } else {
      None
    }

    val g = if (mode.g) {
      // Compute the huber loss gradient elementwise.
      Some(diff.mapElements(diff_i => diff_i / math.max(math.abs(diff_i), tau)))
    } else {
      None
    }

    Value(f, g)
  }
}

/**
 * The log likelihood logistic loss function applied to a DVector.
 *
 * @param y The observed values, labeled as binary 0/1.
 * @param mu The variable values.
 */
class SmoothLogLLogisticDVector(y: DVector) extends SmoothFunction[DVector] with Serializable {

  y.cache()

  override def apply(mu: DVector, mode: Mode): Value[DVector] = {

    val f = if (mode.f) {
      Some(y.zip(mu).treeAggregate(0.0)(
        seqOp = (_, _) match {
          case (sum, (yPart, muPart)) => {

            // Check that the y and mu partitions are the same size.
            if (yPart.size != muPart.size) {
              throw new IllegalArgumentException("Can only zip Vectors with the same number of " +
                "elements")
            }

            yPart.toArray.zip(muPart.toArray).aggregate(0.0)(
              seqop = (sum, elements) => {

                // Compute the loss contribution for each y_i, mu_i pair, and add it to the
                // aggregated sum.
                val (y_i, mu_i) = elements
                val yFactor = if (mu_i > 0.0) y_i - 1.0 else if (mu_i < 0.0) y_i else 0.0
                sum + yFactor * mu_i - math.log1p(math.exp(-math.abs(mu_i)))
              },
              combop = _ + _)
          }
        },
        combOp = _ + _))
    } else {
      None
    }

    val g = if (mode.g) {
      Some(y.zipElements(mu, (y_i, mu_i) => {
        // Compute the log logistic loss gradient elementwise.
        val muFactor = if (mu_i > 0.0) 1.0 else math.exp(mu_i)
        y_i - muFactor / (1.0 + math.exp(-math.abs(mu_i)))
      }))
    } else {
      None
    }

    Value(f, g)
  }
}
