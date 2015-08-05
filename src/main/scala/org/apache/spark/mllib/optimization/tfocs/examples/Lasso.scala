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

package org.apache.spark.mllib.optimization.tfocs.examples

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.optimization.tfocs.{
  ProductVectorDVector,
  ProxL1Vector,
  SmoothQuadDVector,
  TFOCS
}
import org.apache.spark.mllib.optimization.tfocs.VectorSpace._
import org.apache.spark.rdd.RDD

/** Helper to solve lasso regression problems using the tfocs implementation. */
object SolverLasso {

  /**
   * Run a lasso solver on the provided data, using the tfocs implementation.
   *
   * @param A The design matrix, represented as a DMatrix.
   * @param b The observed values, represented as a DVector.
   * @param lambda The regularization term.
   * @param x0 The starting weights.
   *
   * @return The optimized weights returned by the solver.
   */
  def run(A: DMatrix, b: DVector, lambda: Double, x0: Vector): Vector =
    TFOCS.optimize(new SmoothQuadDVector(b),
      new ProductVectorDVector(A),
      new ProxL1Vector(lambda),
      x0)._1
}
