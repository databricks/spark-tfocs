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
import org.apache.spark.mllib.optimization.tfocs.fs.vector.dvector.LinopMatrix
import org.apache.spark.mllib.optimization.tfocs.fs.dvector.vector.LinopMatrixAdjoint
import org.apache.spark.mllib.optimization.tfocs.fs.vector.dvectordouble.{ LinopMatrix => LinopMatrixVector }
import org.apache.spark.mllib.optimization.tfocs.fs.dvectordouble.vector.{ LinopMatrixAdjoint => LinopMatrixVectorAdjoint }
import org.apache.spark.mllib.util.MLlibTestSparkContext

class LinearOperatorSuite extends FunSuite with MLlibTestSparkContext {

  lazy val matrix = sc.parallelize(Array(Vectors.dense(1.0, 2.0, 3.0),
    Vectors.dense(4.0, 5.0, 6.0)), 2)

  lazy val vector = new DenseVector(Array(2.2, 3.3, 4.4))

  test("LinopMatrix multiplies properly") {

    val f = new LinopMatrix(matrix)
    val x = new DenseVector(Array(7.0, 8.0, 9.0))
    val result = f(x)
    val expectedResult = Vectors.dense(1 * 7 + 2 * 8 + 3 * 9, 4 * 7 + 5 * 8 + 6 * 9)
    assert(Vectors.dense(result.collectElements) == expectedResult,
      "should return the correct product")
  }

  test("LinopMatrixAdjoint multiplies properly") {

    val f = new LinopMatrixAdjoint(matrix)
    val y = sc.parallelize(Array(new DenseVector(Array(5.0)), new DenseVector(Array(6.0))), 2)
    val result = f(y)
    val expectedResult = Vectors.dense(1 * 5 + 4 * 6, 2 * 5 + 5 * 6, 3 * 5 + 6 * 6)
    assert(result == expectedResult, "should return the correct product")
  }

  test("LinopMatrixAdjoint checks for mismatched partition vectors") {

    val f = new LinopMatrixAdjoint(matrix)
    val y = sc.parallelize(Array(new DenseVector(Array(5.0, 6.0)), Vectors.zeros(0).toDense), 2)
    intercept[SparkException] {
      f(y)
    }
  }

  test("LinopMatrixVector multiplies properly") {

    val f = new LinopMatrixVector(matrix, vector)
    val x = new DenseVector(Array(7.0, 8.0, 9.0))
    val result = f(x)
    val expectedResult = (new DenseVector(Array(1 * 7 + 2 * 8 + 3 * 9, 4 * 7 + 5 * 8 + 6 * 9)),
      7.0 * 2.2 + 8.0 * 3.3 + 9.0 * 4.4)
    assert(Vectors.dense(result._1.collectElements) == expectedResult._1,
      "should return the correct product")
    assert(result._2 == expectedResult._2, "should return the correct product")
  }

  test("LinopMatrixVectorAdjoint multiplies properly") {

    var f = new LinopMatrixVectorAdjoint(matrix, vector)
    val y = (sc.parallelize(Array(new DenseVector(Array(5.0)), new DenseVector(Array(6.0))), 2),
      8.8)
    val result = f(y)
    val expectedResult =
      Vectors.dense(1 * 5 + 4 * 6 + 2.2, 2 * 5 + 5 * 6 + 3.3, 3 * 5 + 6 * 6 + 4.4)
    assert(result == expectedResult, "should return the correct product")
  }

  test("LinopMatrixVectorAdjoint checks for mismatched partition vectors") {

    val f = new LinopMatrixVectorAdjoint(matrix, vector)
    val y = (sc.parallelize(Array(new DenseVector(Array(5.0, 6.0)), Vectors.zeros(0).toDense), 2),
      8.8)
    intercept[SparkException] {
      f(y)
    }
  }
}
