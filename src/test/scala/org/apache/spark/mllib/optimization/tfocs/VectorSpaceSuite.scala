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

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.mllib.optimization.tfocs.VectorSpace._

class VectorSpaceSuite extends FunSuite with MLlibTestSparkContext {

  test("SimpleVectorSpace.combine is implemented properly") {
    val alpha = 1.1
    val a = Vectors.dense(2.0, 3.0)
    val beta = 4.0
    val b = Vectors.dense(5.0, 6.0)
    val expectedCombination = Vectors.dense(1.1 * 2.0 + 4.0 * 5.0, 1.1 * 3.0 + 4.0 * 6.0)
    assert(SimpleVectorSpace.combine(alpha, a, beta, b) == expectedCombination,
      "SimpleVectorSpace.combine should return the correct result.")
  }

  test("SimpleVectorSpace.dot is implemented properly") {
    val a = Vectors.dense(2.0, 3.0)
    val b = Vectors.dense(5.0, 6.0)
    val expectedDot = 2.0 * 5.0 + 3.0 * 6.0
    assert(SimpleVectorSpace.dot(a, b) == expectedDot,
      "SimpleVectorSpace.dot should return the correct result.")
  }

  test("DVectorVectorSpace.combine is implemented properly") {
    val alpha = 1.1
    val a = sc.parallelize(Array(Vectors.dense(2.0, 3.0), Vectors.dense(4.0)), 2)
    val beta = 4.0
    val b = sc.parallelize(Array(Vectors.dense(5.0, 6.0), Vectors.dense(7.0)), 2)
    val expectedCombination =
      Vectors.dense(1.1 * 2.0 + 4.0 * 5.0, 1.1 * 3.0 + 4.0 * 6.0, 1.1 * 4.0 + 4.0 * 7.0)
    assert(Vectors.dense(DVectorVectorSpace.combine(alpha, a, beta, b).flatMap(_.toArray).toArray)
      == expectedCombination,
      "DVectorVectorSpace.combine should return the correct result.")
  }

  test("DVectorVectorSpace.dot is implemented properly") {
    val a = sc.parallelize(Array(Vectors.dense(2.0, 3.0), Vectors.dense(4.0)), 2)
    val b = sc.parallelize(Array(Vectors.dense(5.0, 6.0), Vectors.dense(7.0)), 2)
    val expectedDot = 2.0 * 5.0 + 3.0 * 6.0 + 4.0 * 7.0
    assert(DVectorVectorSpace.dot(a, b) == expectedDot,
      "DVectorVectorSpace.dot should return the correct result.")
  }
}
