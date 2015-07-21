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

class SmoothFunctionSuite extends FunSuite with MLlibTestSparkContext {

  test("The SmoothQuadRDDDouble implementation should return the expected value and gradient") {

    val x0 = sc.parallelize(Array(1.0, 2.0, 3.0))
    val x = sc.parallelize(Array(10.0, 20.0, 30.0))

    val Value(Some(f), Some(g)) = new SmoothQuadRDDDouble(x0)(x, Mode(true, true))

    assert(f == (math.pow(10 - 1, 2) + math.pow(20 - 2, 2) + math.pow(30 - 3, 2)) / 2,
      "function value should be correct")

    assert(g.collect().deep == Array(10.0 - 1.0, 20.0 - 2.0, 30.0 - 3.0).deep,
      "function gradient should be correct")
  }

  test("The SmoothQuadRDDVector implementation should return the expected value and gradient") {

    val x0 = sc.parallelize(Array(Vectors.dense(1.0, 2.0), Vectors.dense(3.0)), 2)
    val x = sc.parallelize(Array(Vectors.dense(10.0, 20.0), Vectors.dense(30.0)), 2)

    val Value(Some(f), Some(g)) = new SmoothQuadRDDVector(x0)(x, Mode(true, true))

    assert(f == (math.pow(10 - 1, 2) + math.pow(20 - 2, 2) + math.pow(30 - 3, 2)) / 2,
      "function value should be correct")

    assert(g.flatMap(_.toArray).collect().deep == Array(10.0 - 1.0, 20.0 - 2.0, 30.0 - 3.0).deep,
      "function gradient should be correct")
  }

  test("The SmoothQuadRDDVector checks for mismatched partition vectors") {

    val x0 = sc.parallelize(Array(Vectors.dense(1.0), Vectors.dense(2.0, 3.0)), 2)
    val x = sc.parallelize(Array(Vectors.dense(10.0, 20.0), Vectors.dense(30.0)), 2)

    intercept[SparkException] {
      new SmoothQuadRDDVector(x0)(x, Mode(true, true))
    }
  }
}
