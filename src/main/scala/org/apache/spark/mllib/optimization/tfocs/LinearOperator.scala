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
import org.apache.spark.mllib.optimization.tfocs.CheckedIteratorFunctions._
import org.apache.spark.mllib.optimization.tfocs.VectorSpace._

/**
 * A trait for linear operators supporting application of an operator and of its adjoint.
 *
 * @tparam X Type representing an input vector.
 * @tparam Y Type representing an output vector.
 */
trait LinearOperator[X, Y] {
  /**
   * Evaluates this operator at x.
   */
  def apply(x: X): Y

  /**
   * The adjoint of this operator.
   */
  def a: LinearOperator[Y, X]
}

/**
 * Compute the product of a DMatrix with a Vector to produce a DVector.
 *
 * NOTE In matlab tfocs this functionality is implemented in linop_matrix.m.
 */
class ProductVectorDVector(private val matrix: DMatrix)
    extends LinearOperator[Vector, DVector] {

  matrix.cache()

  override def apply(x: Vector): DVector = {
    val bcX = matrix.context.broadcast(x)
    // Take the dot product of each matrix row with x.
    matrix.mapPartitions(partitionRows =>
      Iterator.single(new DenseVector(partitionRows.map(row => BLAS.dot(row, bcX.value)).toArray)))
  }

  override def a: LinearOperator[DVector, Vector] = new AdjointProductVectorDVector(matrix)
}

/**
 * Compute the adjoint product of a DMatrix with a DVector to produce a Vector.
 *
 * The implementation multiplies each row of 'matrix' by the corresponding value of the column
 * vector 'x' and sums the scaled vectors thus obtained.
 *
 * NOTE In matlab tfocs this functionality is implemented in linop_matrix.m.
 */
class AdjointProductVectorDVector(@transient private val matrix: DMatrix)
    extends LinearOperator[DVector, Vector] with java.io.Serializable {

  matrix.cache()

  private lazy val n = matrix.first.size

  override def apply(x: DVector): Vector = {
    matrix.zipPartitions(x)({ (matrixPartition, xPartition) =>
      Iterator.single(
        matrixPartition.checkedZip(xPartition.next.toArray.toIterator).aggregate(Vectors.zeros(n))(
          seqop = (_, _) match {
            case (sum, (matrix_i, x_i)) => {
              // Multiply an element of x by its corresponding matrix row, and add to the running
              // sum vector.
              BLAS.axpy(x_i, matrix_i, sum)
              sum
            }
          },
          combop = (sum1, sum2) => {
            // Add the intermediate sum vectors.
            BLAS.axpy(1.0, sum2, sum1)
            sum1
          }
        ))
    }).treeAggregate(Vectors.zeros(n))(
      seqOp = (sum1, sum2) => {
        // Add the intermediate sum vectors.
        BLAS.axpy(1.0, sum2, sum1)
        sum1
      },
      combOp = (sum1, sum2) => {
        // Add the intermediate sum vectors.
        BLAS.axpy(1.0, sum2, sum1)
        sum1
      }
    )
  }

  override def a: LinearOperator[Vector, DVector] = new ProductVectorDVector(matrix)
}