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

/** Helper to solve lasso regression problems using the tfocs implementation. */
object SolverLasso {

  /**
   * Run a lasso solver on the provided data, using the tfocs implementation.
   *
   * @param A The design matrix, an represented as an RDD of row vectors.
   * @param b The observed values, a column vector of doubles represented as an RDD[Vector]. The
   *        double values within each partition are collected into Vectors for improved performance.
   *        The values of b must be co-partitioned with those of A.
   * @param lambda The regularization term.
   * @param x0 The starting weights.
   *
   * @return The optimized weights returned by the solver.
   */
  def run(A: RDD[Vector], b: RDD[Vector], lambda: Double, x0: Vector): Vector =
    TFOCS.optimize(new SmoothQuadRDDVector(b),
      new ProductVectorRDDVector(A),
      new ProxL1Vector(lambda),
      x0)._1
}
