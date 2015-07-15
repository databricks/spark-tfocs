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

import org.scalatest.{ FunSuite, Matchers }

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLlibTestSparkContext

class ProxCapableFunctionSuite extends FunSuite with Matchers {

  test("The ProxZeroVector implementation should return the expected value and vector") {
    val x = Vectors.dense(10.0, -20.0, 30.0)
    assert(new ProxZeroVector()(x) == 0.0, "value should be correct")
    val Value(Some(f), Some(g)) = new ProxZeroVector()(x, 1.5, Mode(true, true))
    assert(f == 0.0, "minimum value should be correct")
    assert(g == Vectors.dense(10.0, -20.0, 30.0), "minimizing value should be correct")
  }

  test("The ProxL1Vector implementation should return the expected value and vector") {
    val x = Vectors.dense(10.0, -20.0, 30.0)
    assert(new ProxL1Vector(1.1)(x) == 66.0, "value should be correct")
    val Value(Some(f), Some(g)) = new ProxL1Vector(1.1)(x, 1.5, Mode(true, true))
    assert(f == 60.555, "minimum value should be correct")
    assert(g == Vectors.dense(8.35, -18.35, 28.349999999999998),
      "minimizing value should be correct")
  }
}
