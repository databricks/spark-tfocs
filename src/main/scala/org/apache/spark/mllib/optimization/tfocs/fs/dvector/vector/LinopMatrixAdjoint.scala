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

package org.apache.spark.mllib.optimization.tfocs.fs.dvector.vector

import org.apache.spark.mllib.linalg.BLAS
import org.apache.spark.mllib.linalg.{ DenseVector, Vectors }
import org.apache.spark.mllib.optimization.tfocs.CheckedIteratorFunctions._
import org.apache.spark.mllib.optimization.tfocs.fs.vector.dvector.LinopMatrix
import org.apache.spark.mllib.optimization.tfocs.LinearOperator
import org.apache.spark.mllib.optimization.tfocs.VectorSpace._
import org.apache.spark.storage.StorageLevel

/**
 * Compute the adjoint product of a DMatrix with a DVector to produce a Vector.
 *
 * The implementation multiplies each row of 'matrix' by the corresponding value of the column
 * vector 'x' and sums the scaled vectors thus obtained.
 *
 * NOTE In matlab tfocs this functionality is implemented in linop_matrix.m.
 * @see [[https://github.com/cvxr/TFOCS/blob/master/linop_matrix.m]]
 */
class LinopMatrixAdjoint(@transient private val matrix: DMatrix)
    extends LinearOperator[DVector, DenseVector] {

  if (matrix.getStorageLevel == StorageLevel.NONE) {
    matrix.cache()
  }

  private lazy val n = matrix.first().size

  override def apply(x: DVector): DenseVector = {
    val n = this.n
    matrix.zipPartitions(x)((matrixPartition, xPartition) =>
      Iterator.single(
        matrixPartition.checkedZip(xPartition.next.values.toIterator).aggregate(
          // NOTE A DenseVector result is assumed here (not sparse safe).
          Vectors.zeros(n).toDense)(
            seqop = (_, _) match {
              case (sum, (matrix_i, x_i)) => {
                // Multiply an element of x by its corresponding matrix row, and add to the
                // accumulation sum vector.
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
    ).treeAggregate(Vectors.zeros(n).toDense)(
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

  override def t: LinearOperator[DenseVector, DVector] = new LinopMatrix(matrix)
}
