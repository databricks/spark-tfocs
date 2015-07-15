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
import org.apache.spark.mllib.linalg.BLAS
import org.apache.spark.rdd.RDD

/**
 * Trait for a vector space.
 *
 * @tparam X Type representing a vector.
 */
trait VectorSpace[X] {

  /** Linear combination of two vectors. */
  def combine(alpha: Double, a: X, beta: Double, b: X): X

  /** Inner product of two vectors. */
  def dot(a: X, b: X): Double

  /** Cache a vector. */
  def cache(a: X): Unit = {}
}

object VectorSpace {

  /** A VectorSpace for Vectors in local memory. */
  implicit object SimpleVectorSpace extends VectorSpace[Vector] {

    override def combine(alpha: Double, a: Vector, beta: Double, b: Vector): Vector = {
      val ret = a.copy
      if (alpha != 1.0) BLAS.scal(alpha, ret)
      BLAS.axpy(beta, b, ret)
      ret
    }

    override def dot(a: Vector, b: Vector): Double = BLAS.dot(a, b)
  }

  /** A VectorSpace for RDD[Double] vectors. */
  implicit object RDDDoubleVectorSpace extends VectorSpace[RDD[Double]] {

    override def combine(alpha: Double, a: RDD[Double], beta: Double, b: RDD[Double]): RDD[Double] =
      a.zip(b).map(x => alpha * x._1 + beta * x._2)

    override def dot(a: RDD[Double], b: RDD[Double]): Double =
      a.zip(b).treeAggregate(0.0)((sum, x) => sum + x._1 * x._2, _ + _)

    override def cache(a: RDD[Double]): Unit = a.cache()
  }

  /** A VectorSpace for RDD[Vector] vectors. */
  implicit object RDDVectorVectorSpace extends VectorSpace[RDD[Vector]] {

    override def combine(alpha: Double, a: RDD[Vector], beta: Double, b: RDD[Vector]): RDD[Vector] =
      a.zip(b).map(x =>
        new DenseVector(x._1.toArray.zip(x._2.toArray).map(y =>
          alpha * y._1 + beta * y._2)): Vector)

    override def dot(a: RDD[Vector], b: RDD[Vector]): Double =
      a.zip(b).map(x => BLAS.dot(x._1, x._2)).sum

    override def cache(a: RDD[Vector]): Unit = a.cache()
  }

}
