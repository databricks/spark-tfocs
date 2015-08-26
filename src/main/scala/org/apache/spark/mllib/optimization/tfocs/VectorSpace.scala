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

import org.apache.spark.mllib.linalg.{ DenseVector, Vector }
import org.apache.spark.rdd.RDD

/**
 * A trait for a vector space supporting a few basic linear algebra operations.
 *
 * @tparam X A type representing a vector.
 */
trait VectorSpace[X] {

  /** Compute a linear combination of two vectors. */
  def combine(alpha: Double, a: X, beta: Double, b: X): X

  /** Compute the inner product of two vectors. */
  def dot(a: X, b: X): Double

  /** Cache a vector. */
  def cache(a: X): Unit = {}
}

object VectorSpace {

  /**
   * A distributed one dimensional vector stored as an RDD of mllib.linalg DenseVectors, where each
   * RDD partition contains a single DenseVector. This representation provides improved performance
   * over RDD[Double], which requires that each element be unboxed during elementwise operations.
   */
  type DVector = RDD[DenseVector]

  /**
   * A distributed two dimensional matrix stored as an RDD of mllib.linalg Vectors, where each
   * Vector represents a row of the matrix. The Vectors may be dense or sparse.
   *
   * NOTE In order to multiply the transpose of a DMatrix 'm' by a DVector 'v', m and v must be
   * consistently partitioned. Each partition of m must contain the same number of rows as there
   * are vector elements in the corresponding partition of v. For example, if m contains two
   * partitions and there are two row Vectors in the first partition and three row Vectors in the
   * second partition, then v must have two partitions with a single Vector containing two elements
   * in its first partition and a single Vector containing three elements in its second partition.
   */
  type DMatrix = RDD[Vector]
}
