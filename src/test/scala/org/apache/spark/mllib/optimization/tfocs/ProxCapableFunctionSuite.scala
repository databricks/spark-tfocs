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

import org.apache.spark.mllib.linalg.{ BLAS, DenseVector, Vectors }
import org.apache.spark.mllib.optimization.tfocs.DVectorFunctions._
import org.apache.spark.mllib.optimization.tfocs.fs.vector.double._
import org.apache.spark.mllib.optimization.tfocs.fs.dvector.double._
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.mllib.util.TestingUtils._

class ProxCapableFunctionSuite extends FunSuite with MLlibTestSparkContext {

  test("The ProxZero implementation should return the expected value and vector") {
    val fun = new ProxZero()
    val x = new DenseVector(Array(10.0, -20.0, 30.0))
    assert(fun(x) == 0.0, "value should be correct")
    val ProxValue(Some(f), Some(g)) = fun(x, 1.5, ProxMode(true, true))
    assert(f == 0.0, "minimum value should be correct")
    assert(g == Vectors.dense(10.0, -20.0, 30.0), "minimizing value should be correct")
  }

  test("The ProxL1 implementation should return the expected value and vector") {
    val fun = new ProxL1(1.1)
    val x = new DenseVector(Array(10.0, -20.0, 30.0))
    assert(fun(x) == 66.0, "value should be correct")
    val ProxValue(Some(f), Some(g)) = fun(x, 1.5, ProxMode(true, true))
    assert(f ~= 60.555 relTol 1e-12, "minimum value should be correct")
    assert(g ~= Vectors.dense(8.35, -18.35, 28.34999999) relTol 1e-6,
      "minimizing value should be correct")
  }

  test("The ProjRPlus implementation should return the expected value and vector") {

    // Already nonnegative.
    val fun = new ProjRPlus()
    val x1 = new DenseVector(Array(10.0, 20.0, 30.0))
    val ProxValue(Some(f1), Some(g1)) = fun(x1, 1.0, ProxMode(true, true))
    assert(f1 == 0.0, "value should be correct")
    assert(g1 == x1, "vector should be correct")
    assert(fun(x1) == 0.0,
      "value inside the nonnegative orthant should be correct for function short form")

    // Some negative elements.
    val x2 = new DenseVector(Array(-10.0, 20.0, -30.0))
    val ProxValue(Some(f2), Some(g2)) = fun(x2, 1.0, ProxMode(true, true))
    assert(f2 == 0.0, "value should be correct")
    assert(g2 == Vectors.dense(0.0, 20.0, 0.0), "vector should be correct")
    assert(fun(x2) == Double.PositiveInfinity,
      "value outisde the nonnegative orthant should be correct for function short form")
  }

  test("The ProjBox implementation should return the expected value and vector") {

    // Already within box.
    val fun1 = new ProjBox(new DenseVector(Array(9, 19, 29)), new DenseVector(Array(11, 21, 31)))
    val x1 = new DenseVector(Array(10.0, 20.0, 30.0))
    val ProxValue(Some(f1), Some(g1)) = fun1(x1, 1.0, ProxMode(true, true))
    assert(f1 == 0.0, "value should be correct")
    assert(g1 == x1, "vector should be correct")
    assert(fun1(x1) == 0.0, "value within the box should be correct for function short form")

    // Some elements outside box.
    val fun2 = new ProjBox(new DenseVector(Array(10.5, 19, 29)),
      new DenseVector(Array(11, 21, 29.5)))
    val x2 = new DenseVector(Array(10.0, 20.0, 30.0))
    val ProxValue(Some(f2), Some(g2)) = fun2(x2, 1.0, ProxMode(true, true))
    assert(f2 == 0.0, "value should be correct")
    assert(g2 == Vectors.dense(10.5, 20, 29.5), "vector should be correct")

    // Some elements outside other boxes.
    val fun3 = new ProjBox(new DenseVector(Array(10.5, 19, 29)), new DenseVector(Array(11, 21, 31)))
    assert(fun3(x2) == Double.PositiveInfinity,
      "value outisde the box should be correct for function short form")
    val fun4 = new ProjBox(new DenseVector(Array(10, 19, 29)), new DenseVector(Array(11, 21, 29.5)))
    assert(fun4(x2) == Double.PositiveInfinity,
      "value outisde the box should be correct for function short form")
  }

  test("The ProxShiftPlus implementation should return the expected value and vector") {

    // Already nonnegative.
    val c1 = sc.parallelize(Array(new DenseVector(Array(9.0, 19.0)), new DenseVector(Array(29.0))))
    val fun1 = new ProxShiftRPlus(c1)
    val x1 = sc.parallelize(Array(new DenseVector(Array(10.0, 20.0)), new DenseVector(Array(30.0))))
    val expectedEvalF1 = 10 * 9 + 19 * 20 + 29 * 30
    assert(fun1(x1) == expectedEvalF1, "eval value should be correct")
    val ProxValue(Some(f1), Some(g1)) = fun1(x1, 0.8, ProxMode(true, true))
    val expectedG1 = Vectors.dense(10 - .8 * 9, 20 - .8 * 19, 30 - .8 * 29)
    val expectedF1 = BLAS.dot(Vectors.dense(c1.flatMap(_.toArray).collect), expectedG1)
    assert(f1 == expectedF1, "value should be correct")
    assert(Vectors.dense(g1.collectElements) == expectedG1, "vector should be correct")

    // Some negative elements.
    val c2 = sc.parallelize(Array(new DenseVector(Array(9.0, -19.0)),
      new DenseVector(Array(-29.0))))
    val fun2 = new ProxShiftRPlus(c2)
    val x2 = sc.parallelize(Array(new DenseVector(Array(-10.0, 20.0)),
      new DenseVector(Array(-30.0))))
    assert(fun2(x2) == Double.PositiveInfinity, "eval value should be correct")
    val ProxValue(Some(f2), Some(g2)) = fun2(x2, 0.8, ProxMode(true, true))
    val expectedG2 = Vectors.dense(0.0, 20 - .8 * -19, 0.0)
    val expectedF2 = BLAS.dot(Vectors.dense(c2.flatMap(_.toArray).collect), expectedG2)
    assert(f2 == expectedF2, "value should be correct")
    assert(Vectors.dense(g2.collectElements) == expectedG2, "vector should be correct")
  }
}
