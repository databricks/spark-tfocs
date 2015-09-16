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

import org.apache.spark.mllib.linalg.{ DenseVector, Vectors }
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.mllib.optimization.tfocs.DVectorFunctions._
import org.apache.spark.mllib.optimization.tfocs.VectorSpace._
import org.apache.spark.mllib.optimization.tfocs.vs.dvector.DVectorSpace
import org.apache.spark.mllib.optimization.tfocs.vs.dvectordouble.DVectorDoubleSpace
import org.apache.spark.mllib.optimization.tfocs.vs.vector.DenseVectorSpace

class VectorSpaceSuite extends FunSuite with MLlibTestSparkContext {

  test("DenseVectorSpace.combine is implemented properly") {
    val alpha = 1.1
    val a = new DenseVector(Array(2.0, 3.0))
    val beta = 4.0
    val b = new DenseVector(Array(5.0, 6.0))
    val expectedCombination = Vectors.dense(1.1 * 2.0 + 4.0 * 5.0, 1.1 * 3.0 + 4.0 * 6.0)
    assert(DenseVectorSpace.combine(alpha, a, beta, b) == expectedCombination,
      "DenseVectorSpace.combine should return the correct result.")
  }

  test("DenseVectorSpace.dot is implemented properly") {
    val a = new DenseVector(Array(2.0, 3.0))
    val b = new DenseVector(Array(5.0, 6.0))
    val expectedDot = 2.0 * 5.0 + 3.0 * 6.0
    assert(DenseVectorSpace.dot(a, b) == expectedDot,
      "DenseVectorSpace.dot should return the correct result.")
  }

  test("DVectorSpace.combine is implemented properly") {
    val alpha = 1.1
    val a = sc.parallelize(Array(new DenseVector(Array(2.0, 3.0)), new DenseVector(Array(4.0))), 2)
    val beta = 4.0
    val b = sc.parallelize(Array(new DenseVector(Array(5.0, 6.0)), new DenseVector(Array(7.0))), 2)
    val combination = DVectorSpace.combine(alpha, a, beta, b)
    val expectedCombination =
      Vectors.dense(1.1 * 2.0 + 4.0 * 5.0, 1.1 * 3.0 + 4.0 * 6.0, 1.1 * 4.0 + 4.0 * 7.0)
    assert(Vectors.dense(combination.collectElements) == expectedCombination,
      "DVectorSpace.combine should return the correct result.")
  }

  test("DVectorSpace.dot is implemented properly") {
    val a = sc.parallelize(Array(new DenseVector(Array(2.0, 3.0)), new DenseVector(Array(4.0))), 2)
    val b = sc.parallelize(Array(new DenseVector(Array(5.0, 6.0)), new DenseVector(Array(7.0))), 2)
    val expectedDot = 2.0 * 5.0 + 3.0 * 6.0 + 4.0 * 7.0
    assert(DVectorSpace.dot(a, b) == expectedDot,
      "DVectorSpace.dot should return the correct result.")
  }

  test("DVectorDoubleSpace.combine is implemented properly") {
    val alpha = 1.1
    val a = (sc.parallelize(Array(new DenseVector(Array(2.0, 3.0)), new DenseVector(Array(4.0))),
      2), 9.9)
    val beta = 4.0
    val b = (sc.parallelize(Array(new DenseVector(Array(5.0, 6.0)), new DenseVector(Array(7.0))),
      2), 11.11)
    val combination = DVectorDoubleSpace.combine(alpha, a, beta, b)
    val expectedCombination =
      (Vectors.dense(1.1 * 2.0 + 4.0 * 5.0, 1.1 * 3.0 + 4.0 * 6.0, 1.1 * 4.0 + 4.0 * 7.0),
        1.1 * 9.9 + 4.0 * 11.11)
    assert(Vectors.dense(combination._1.collectElements) == expectedCombination._1,
      "DVectorVectorSpace.combine should return the correct result.")
    assert(combination._2 == expectedCombination._2,
      "DVectorVectorSpace.combine should return the correct result.")
  }

  test("DVectorDoubleSpace.dot is implemented properly") {
    val a = (sc.parallelize(Array(new DenseVector(Array(2.0, 3.0)), new DenseVector(Array(4.0))),
      2), 9.9)
    val b = (sc.parallelize(Array(new DenseVector(Array(5.0, 6.0)), new DenseVector(Array(7.0))),
      2), 11.11)
    val expectedDot = 2.0 * 5.0 + 3.0 * 6.0 + 4.0 * 7.0 + 9.9 * 11.11
    assert(DVectorDoubleSpace.dot(a, b) == expectedDot,
      "DVectorVectorSpace.dot should return the correct result.")
  }
}
