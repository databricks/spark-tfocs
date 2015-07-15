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

class VectorSpaceSuite extends FunSuite with MLlibTestSparkContext with Matchers {

  test("SimpleVectorSpace is implemented properly") {
    assert(Vectors.dense(22.2, 27.3) ==
      VectorSpace.SimpleVectorSpace.combine(1.1, Vectors.dense(2.0, 3.0),
        4.0, Vectors.dense(5.0, 6.0)),
      "SimpleVectorSpace.combine should return the correct result.")

    assert(28 ==
      VectorSpace.SimpleVectorSpace.dot(Vectors.dense(2.0, 3.0), Vectors.dense(5.0, 6.0)),
      "SimpleVectorSpace.dot should return the correct result.")
  }

  test("RDDDoubleVectorSpace is implemented properly") {
    assert(Array(22.2, 27.3).deep ==
      VectorSpace.RDDDoubleVectorSpace.combine(1.1, sc.parallelize(Array(2.0, 3.0)),
        4.0, sc.parallelize(Array(5.0, 6.0))).collect().deep,
      "RDDDoubleVectorSpace.combine should return the correct result.")

    assert(28 ==
      VectorSpace.RDDDoubleVectorSpace.dot(sc.parallelize(Array(2.0, 3.0)),
        sc.parallelize(Array(5.0, 6.0))),
      "RDDDoubleVectorSpace.dot should return the correct result.")
  }

  test("RDDVectorVectorSpace is implemented properly") {
    assert(Array(22.2, 27.3, 32.4).deep ==
      VectorSpace.RDDVectorVectorSpace.combine(1.1,
        sc.parallelize(Array(Vectors.dense(2.0, 3.0), Vectors.dense(4.0)), 2),
        4.0,
        sc.parallelize(Array(Vectors.dense(5.0, 6.0), Vectors.dense(7.0)), 2))
      .flatMap(_.toArray).collect().deep,
      "RDDVectorVectorSpace.combine should return the correct result.")

    assert(56 ==
      VectorSpace.RDDVectorVectorSpace.dot(
        sc.parallelize(Array(Vectors.dense(2.0, 3.0), Vectors.dense(4.0)), 2),
        sc.parallelize(Array(Vectors.dense(5.0, 6.0), Vectors.dense(7.0)), 2)),
      "RDDVectorVectorSpace.dot should return the correct result.")
  }
}
