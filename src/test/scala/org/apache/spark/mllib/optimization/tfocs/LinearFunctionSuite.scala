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
import org.apache.spark.mllib.util.MLlibTestSparkContext

class LinearFunctionSuite extends FunSuite with MLlibTestSparkContext {

  lazy val matrix = sc.parallelize(Array(Vectors.dense(1.0, 2.0, 3.0),
    Vectors.dense(4.0, 5.0, 6.0)), 2)

  test("ProductVectorDVector multiplies properly") {

    val f = new ProductVectorDVector(matrix)
    val x = Vectors.dense(7.0, 8.0, 9.0)
    var result = f(x)
    assert(Vectors.dense(result.flatMap(_.toArray).collect()) == Vectors.dense(50.0, 122.0),
      "should return the correct product")
  }

  test("TransposeProductVectorDVector multiplies properly") {

    var f = new TransposeProductVectorDVector(matrix)
    val y = sc.parallelize(Array(Vectors.dense(5.0), Vectors.dense(6.0)), 2)
    var result = f(y)
    assert(result == Vectors.dense(29.0, 40.0, 51.0), "should return the correct product")
  }

  test("TransposeProductVectorDVector checks for mismatched partition vectors") {

    val f = new TransposeProductVectorDVector(matrix)
    val y = sc.parallelize(Array(Vectors.dense(5.0, 6.0), Vectors.zeros(0)), 2)
    intercept[SparkException] {
      f(y)
    }
  }
}
