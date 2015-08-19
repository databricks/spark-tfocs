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
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.tfocs.DVectorFunctions._
import org.apache.spark.mllib.util.MLlibTestSparkContext

class LinearOperatorSuite extends FunSuite with MLlibTestSparkContext {

  lazy val matrix = sc.parallelize(Array(Vectors.dense(1.0, 2.0, 3.0),
    Vectors.dense(4.0, 5.0, 6.0)), 2)

  test("LinopMatrix multiplies properly") {

    val f = new LinopMatrix(matrix)
    val x = Vectors.dense(7.0, 8.0, 9.0)
    val result = f(x)
    val expectedResult = Vectors.dense(1 * 7 + 2 * 8 + 3 * 9, 4 * 7 + 5 * 8 + 6 * 9)
    assert(Vectors.dense(result.collectElements) == expectedResult,
      "should return the correct product")
  }

  test("LinopMatrixAdjoint multiplies properly") {

    val f = new LinopMatrixAdjoint(matrix)
    val y = sc.parallelize(Array(Vectors.dense(5.0), Vectors.dense(6.0)), 2)
    val result = f(y)
    val expectedResult = Vectors.dense(1 * 5 + 4 * 6, 2 * 5 + 5 * 6, 3 * 5 + 6 * 6)
    assert(result == expectedResult, "should return the correct product")
  }

  test("LinopMatrixAdjoint checks for mismatched partition vectors") {

    val f = new LinopMatrixAdjoint(matrix)
    val y = sc.parallelize(Array(Vectors.dense(5.0, 6.0), Vectors.zeros(0)), 2)
    intercept[SparkException] {
      f(y)
    }
  }
}
