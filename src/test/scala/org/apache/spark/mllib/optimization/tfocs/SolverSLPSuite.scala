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

import org.scalatest.FunSuite
import scala.io.Source

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.tfocs.DVectorFunctions._
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.mllib.util.TestingUtils._

class SolverSLPSuite extends FunSuite with MLlibTestSparkContext {

  test("The values and losses returned by Spark SolverSLP should match those returned by Matlab " +
    "tfocs's solver_sLP") {

    // The test below checks that the results match those of the following TFOCS matlab program
    // (using TFOCS version 1945a771f315acd4cc6eba638b5c01fb52ee7aaa):
    //
    // A = sparse([2, 5, 3, 2, 4], [2, 2, 3, 8, 10], [0.632374636716572, 0.198436985375040, ...
    //   0.179885783103202, 0.014792694748719, 0.244326895623829], 5, 10);
    // b = [0 7.127414296861894 1.781441255102280 2.497425876822379 2.186136752456199]';
    // c = [-1.078275146772097 -0.368208440839284 0.680376092886272 0.256371934668609 ...
    //   1.691983132986665 0.059837119884475 -0.221648385883038 -0.298134575377277 ...
    //   -1.913199010346937 0.745084172661387]';
    // mu = 1e-2;
    // opts = struct('restart', -Inf, 'continuation', true, 'noscale', true, ...
    //   'output_always_use_x', true, 'stopCrit', 4, 'tol', 1e-3);
    // contOpts = struct('maxIts', 10, 'initialTol', 1e-2);
    // [x,out,optsOut] = solver_sLP(c, A, b, mu, [], [], opts, contOpts);
    // format long
    // x
    // out.f

    val A = sc.parallelize(Array(Vectors.zeros(5),
      Vectors.sparse(5, Seq((1, 0.632374636716572), (4, 0.198436985375040))),
      Vectors.sparse(5, Seq((2, 0.179885783103202))), Vectors.zeros(5), Vectors.zeros(5),
      Vectors.zeros(5), Vectors.zeros(5), Vectors.sparse(5, Seq((1, 0.014792694748719))),
      Vectors.zeros(5), Vectors.sparse(5, Seq((3, 0.244326895623829)))), 2)
    var b = Vectors.dense(0, 7.127414296861894, 1.781441255102280, 2.497425876822379,
      2.186136752456199).toDense
    val c = sc.parallelize(Array(-1.078275146772097, -0.368208440839284, 0.680376092886272,
      0.256371934668609, 1.691983132986665, 0.059837119884475, -0.221648385883038,
      -0.298134575377277, -1.913199010346937, 0.745084172661387), 2).glom.map(
      Vectors.dense(_).toDense)
    val mu = 0.01
    val x0 = sc.parallelize(Array.fill(10)(0.0), 2).glom.map(Vectors.dense(_).toDense)
    val z0 = Vectors.zeros(5).toDense
    val dualTolCheckInterval = 1 // Matlab tfocs checks for convergence on every iteration.

    val (x, lossHistory) = SolverSLP.run(c, A, b, mu, x0, z0, 10, 1e-3, 1e-2, dualTolCheckInterval)

    val expectedX = Vectors.dense(2048.722778866985, 0, 0, 0, 0, 0, 421.131933177772,
      546.803269626285, 3635.078119659181, 10.514625914138)
    val expectedLossHistory = Array(-252.005414340769, -252.005414340769, -251.156484099887,
      -250.900750472038, -250.441137874951, -746.515181668927, -746.515181668927, -745.988362852497,
      -1365.253694768042, -1365.253694768042, -1364.529817385060, -2107.579888200214,
      -2107.579888200214, -2106.568363677963, -2973.333616393671, -2973.333616393671,
      -2971.953815423126, -3962.641290922493, -3962.641290922493, -3961.901828375015,
      -5076.658876844795, -5076.658876844795, -5076.122659281430, -5075.480196164118,
      -5074.650921295725, -5073.921424808851, -5072.987948040954, -6311.277495149125,
      -6311.277495149125, -6310.451241168823, -7672.322045345107, -7672.322045345107,
      -7671.209444458280, -9157.089180439810, -9157.089180439810, -9155.947984506271)

    assert(Vectors.dense(x.collectElements) ~= expectedX relTol 1e-12,
      "Each x vector element should match the expected value, within tolerance.")

    // The tfocs implementation may return loss values for either the x or y vectors of the
    // accelerated descent implementation. The spark implementation may also return losses for
    // either x or y but using different criteria to select between these vectors. As a result
    // only the first and last losses reported by the optimization task are validated here, both of
    // which are calculated from the x vector.

    assert(lossHistory.length == expectedLossHistory.length,
      "The number of iterations should be the same.")

    assert(lossHistory.head ~= expectedLossHistory.head relTol 1e-12,
      "The loss values on the first iteration should match, within tolerance.")

    assert(lossHistory.last ~= expectedLossHistory.last relTol 1e-12,
      "The loss values on the last iteration should match, within tolerance.")
  }
}
