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

import org.apache.spark.mllib.linalg.BLAS
import org.apache.spark.mllib.linalg.{ DenseVector, Vector, Vectors }
import org.apache.spark.mllib.optimization.tfocs.CheckedIterator._
import org.apache.spark.rdd.RDD

/**
 * Trait for linear functions.
 *
 * @tparam X Type representing a linear function input vector.
 * @tparam Y Type representing a linear function output vector.
 */
trait LinearFunction[X, Y] {
  /**
   * Evaluates this function at x.
   */
  def apply(x: X): Y

  /**
   * The transpose of this function.
   */
  def t: LinearFunction[Y, X]
}

/** Compute the product of an RDD[Vector] matrix with a Vector to produce an RDD[Double] vector. */
class ProductVectorRDDDouble(private val matrix: RDD[Vector])
    extends LinearFunction[Vector, RDD[Double]] {

  matrix.cache()

  override def apply(x: Vector): RDD[Double] = {
    val bcX = matrix.context.broadcast(x)
    matrix.map(row => BLAS.dot(row, bcX.value))
  }

  override def t: LinearFunction[RDD[Double], Vector] = new TransposeProductVectorRDDDouble(matrix)
}

/**
 * Compute the transpose product of an RDD[Vector] matrix with an RDD[Double] vector to produce a
 * Vector.
 */
class TransposeProductVectorRDDDouble(private val matrix: RDD[Vector])
    extends LinearFunction[RDD[Double], Vector] {

  matrix.cache()

  private lazy val n = matrix.first.size

  override def apply(x: RDD[Double]): Vector = {
    matrix.zip(x).treeAggregate(Vectors.zeros(n))(
      seqOp = (sum, row) => {
        BLAS.axpy(row._2, row._1, sum)
        sum
      },
      combOp = (s1, s2) => {
        BLAS.axpy(1.0, s2, s1)
        s1
      }
    )
  }

  override def t: LinearFunction[Vector, RDD[Double]] = new ProductVectorRDDDouble(matrix)
}

/** Compute the product of an RDD[Vector] matrix with a Vector to produce an RDD[Vector] vector. */
class ProductVectorRDDVector(private val matrix: RDD[Vector])
    extends LinearFunction[Vector, RDD[Vector]] {

  matrix.cache()

  override def apply(x: Vector): RDD[Vector] = {
    val bcX = matrix.context.broadcast(x)
    matrix.mapPartitions(rows =>
      Iterator.single(new DenseVector(rows.map(row => BLAS.dot(row, bcX.value)).toArray)))
  }

  override def t: LinearFunction[RDD[Vector], Vector] = new TransposeProductVectorRDDVector(matrix)
}

/**
 * Compute the transpose product of an RDD[Vector] matrix with an RDD[Vector] vector to produce a
 * Vector.
 */
class TransposeProductVectorRDDVector(@transient private val matrix: RDD[Vector])
    extends LinearFunction[RDD[Vector], Vector] with java.io.Serializable {

  matrix.cache()

  private lazy val n = matrix.first.size

  override def apply(x: RDD[Vector]): Vector = {
    matrix.zipPartitions(x)({ (matrixPartition, xPartition) =>
      Iterator.single(
        matrixPartition.checkedZip(xPartition.next.toArray.toIterator).aggregate(Vectors.zeros(n))(
          seqop = (sum, row) => {
            BLAS.axpy(row._2, row._1, sum)
            sum
          },
          combop = (s1, s2) => {
            BLAS.axpy(1.0, s2, s1)
            s1
          }
        ))
    }).treeAggregate(Vectors.zeros(n))(
      seqOp = (s1, s2) => {
        BLAS.axpy(1.0, s2, s1)
        s1
      },
      combOp = (s1, s2) => {
        BLAS.axpy(1.0, s2, s1)
        s1
      }
    )
  }

  override def t: LinearFunction[Vector, RDD[Vector]] = new ProductVectorRDDVector(matrix)
}
