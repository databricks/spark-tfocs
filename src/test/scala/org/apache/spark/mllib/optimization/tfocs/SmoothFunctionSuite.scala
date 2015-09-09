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

import org.apache.spark.SparkException
import org.apache.spark.mllib.linalg.{ DenseVector, Vectors }
import org.apache.spark.mllib.optimization.tfocs.DVectorFunctions._
import org.apache.spark.mllib.optimization.tfocs.fs.dvector.double._
import org.apache.spark.mllib.optimization.tfocs.fs.dvectordouble.double._
import org.apache.spark.mllib.optimization.tfocs.fs.generic.double._
import org.apache.spark.mllib.optimization.tfocs.vs.dvector._
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.mllib.util.TestingUtils._

class SmoothFunctionSuite extends FunSuite with MLlibTestSparkContext {

  test("The SmoothQuad implementation should return the expected value and gradient") {

    val x0 = sc.parallelize(Array(Vectors.dense(1.0, 2.0).toDense, Vectors.dense(3.0).toDense), 2)
    val x = sc.parallelize(Array(Vectors.dense(10.0, 20.0).toDense, Vectors.dense(30.0).toDense), 2)
    val fun = new SmoothQuad(x0)

    val Value(Some(f), Some(g)) = fun(x, Mode(true, true))

    val expectedF = (math.pow(10 - 1, 2) + math.pow(20 - 2, 2) + math.pow(30 - 3, 2)) / 2
    assert(f == expectedF, "function value should be correct")

    val expectedG = Vectors.dense(10.0 - 1.0, 20.0 - 2.0, 30.0 - 3.0)
    assert(Vectors.dense(g.collectElements) == expectedG,
      "function gradient should be correct")
  }

  test("The SmoothQuad implementation checks for mismatched partition vectors") {

    // x0 and x do not have the same number of vector values in each partition.
    val x0 = sc.parallelize(Array(Vectors.dense(1.0).toDense, Vectors.dense(2.0, 3.0).toDense), 2)
    val x = sc.parallelize(Array(Vectors.dense(10.0, 20.0).toDense, Vectors.dense(30.0).toDense), 2)
    val fun = new SmoothQuad(x0)

    intercept[SparkException] {
      fun(x, Mode(true, true))
    }
  }

  test("The SmoothHuber implementation should return the expected value and gradient") {

    val x0 = sc.parallelize(Array(Vectors.dense(1.0, 2.0).toDense,
      Vectors.dense(-3.0, -4.0).toDense), 2)
    val x = sc.parallelize(Array(Vectors.dense(1.1, 1.8).toDense,
      Vectors.dense(-3.3, -3.6).toDense), 2)

    val fun1 = new SmoothHuber(x0, 0.2)
    val Value(Some(f1), Some(g1)) = fun1(x, Mode(true, true))

    val expectedF1 = .5 * .1 * .1 / .2 + .5 * .2 * .2 / .2 + .3 - .2 / 2 + .4 - .2 / 2
    assert(f1 ~= expectedF1 relTol 1e-15,
      "function value should be correct")

    val expectedG1 = Vectors.dense(.1 / .2, -.2 / .2, -.3 / .3, .4 / .4)
    assert(Vectors.dense(g1.collectElements) ~= expectedG1 relTol 1e-15,
      "function gradient should be correct")

    val fun2 = new SmoothHuber(x0, 0.3)
    val Value(Some(f2), Some(g2)) = fun2(x, Mode(true, true))

    val expectedF2 = .5 * .1 * .1 / .3 + .5 * .2 * .2 / .3 + .5 * .3 * .3 / .3 + .4 - .3 / 2
    assert(f2 ~= expectedF2 relTol 1e-15, "function value should be correct")

    val expectedG2 = Vectors.dense(.1 / .3, -.2 / .3, -.3 / .3, .4 / .4)
    assert(Vectors.dense(g2.collectElements) ~= expectedG2 relTol 1e-15,
      "function gradient should be correct")
  }

  test("The SmoothLogLLogistic implementation should return the expected value and gradient") {

    val y = sc.parallelize(Array(Vectors.dense(1.0, 0.0).toDense,
      Vectors.dense(0.0, 1.0, 1.0).toDense), 2)
    val mu = sc.parallelize(Array(Vectors.dense(0.1, -0.2).toDense,
      Vectors.dense(0.3, -0.4, 0.0).toDense), 2)
    val fun = new SmoothLogLLogistic(y)

    val Value(Some(f), Some(g)) = fun(mu, Mode(true, true))

    val expectedF = 0 - math.log1p(math.exp(-0.1)) +
      0 - math.log1p(math.exp(-0.2)) +
      -1 * 0.3 - math.log1p(math.exp(-0.3)) +
      1 * -0.4 - math.log1p(math.exp(-0.4)) +
      0 - math.log1p(math.exp(0))
    assert(f == expectedF, "function value should be correct")

    val expectedG = Vectors.dense(1.0 - 1.0 / (1.0 + math.exp(-0.1)),
      0.0 - math.exp(-0.2) / (1.0 + math.exp(-0.2)),
      0.0 - 1.0 / (1.0 + math.exp(-0.3)),
      1.0 - math.exp(-0.4) / (1.0 + math.exp(-0.4)),
      1.0 - math.exp(0.0) / (1.0 + math.exp(0.0)))
    assert(Vectors.dense(g.collectElements) == expectedG, "function gradient should be correct")
  }

  test("The SmoothDual implementation should return the expected value and gradient") {

    val c = sc.parallelize(Array(Vectors.dense(1.0, -2.2).toDense,
      Vectors.dense(3.4, -4.6, 5.8).toDense), 2)
    val objectiveF = new ProxShiftRPlus(c)
    val mu = 0.2
    val x0 = sc.parallelize(Array(Vectors.dense(2.0, -1.2).toDense,
      Vectors.dense(0.4, -1.6, 2.8).toDense), 2)
    val ATz = sc.parallelize(Array(Vectors.dense(12.0, -11.2).toDense,
      Vectors.dense(10.4, -11.6, 12.8).toDense), 2)
    val fun = new SmoothDual(objectiveF, mu, x0)

    val Value(Some(f), Some(g)) = fun(ATz, Mode(true, true))

    val expectedOffsetCenter = sc.parallelize(Array(Vectors.dense(mu * 12.0 + 2.0,
      mu * -11.2 + -1.2).toDense,
      Vectors.dense(mu * 10.4 + 0.4,
        mu * -11.6 + -1.6,
        mu * 12.8 + 2.8).toDense), 2)
    val ProxValue(Some(expectedProxF), Some(expectedProxMinimizer)) =
      objectiveF(expectedOffsetCenter, mu, ProxMode(true, true))

    val diff = x0.diff(expectedProxMinimizer)
    val expectedF = ATz.dot(expectedProxMinimizer) - expectedProxF -
      (0.5 / mu) * diff.dot(diff)
    assert(f == expectedF, "function value should be correct")

    val expectedG = expectedProxMinimizer.mapElements(g_i => -g_i)
    assert(Vectors.dense(g.collectElements) == Vectors.dense(expectedG.collectElements),
      "function gradient should be correct")
  }

  test("The SmoothCombine implementation should return the expected value and gradient") {

    val x0 = sc.parallelize(Array(Vectors.dense(2.0, -1.2).toDense,
      Vectors.dense(0.4, -1.6, 2.8).toDense), 2)
    val objectiveF = new SmoothQuad(x0)
    val x = (sc.parallelize(Array(Vectors.dense(9.0, 10.1).toDense,
      Vectors.dense(11.2, 12.3, 13.4).toDense), 2),
      88.1)
    val fun = new SmoothCombine(objectiveF)

    val Value(Some(f), Some(g)) = fun(x, Mode(true, true))

    val expectedF = (math.pow(9 - 2, 2) + math.pow(10.1 - -1.2, 2) + math.pow(11.2 - 0.4, 2) +
      math.pow(12.3 - -1.6, 2) + math.pow(13.4 - 2.8, 2)) / 2 + 88.1
    assert(f == expectedF, "function value should be correct")

    val expectedG1 = Vectors.dense(9 - 2, 10.1 - -1.2, 11.2 - 0.4, 12.3 - -1.6, 13.4 - 2.8)
    assert(Vectors.dense(g._1.collectElements) == expectedG1, "function gradient should be correct")

    val expectedG2 = 1.0
    assert(g._2 == expectedG2, "function gradient should be correct")
  }
}
