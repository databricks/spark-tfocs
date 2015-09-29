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

package org.apache.spark.mllib.optimization.tfocs.fs.vector.dvector

import org.apache.spark.mllib.linalg.{ BLAS, DenseVector }
import org.apache.spark.mllib.optimization.tfocs.fs.dvector.vector.LinopMatrixAdjoint
import org.apache.spark.mllib.optimization.tfocs.LinearOperator
import org.apache.spark.mllib.optimization.tfocs.VectorSpace._
import org.apache.spark.storage.StorageLevel

/**
 * Compute the product of a DMatrix with a Vector to produce a DVector.
 *
 * NOTE In matlab tfocs this functionality is implemented in linop_matrix.m.
 * @see [[https://github.com/cvxr/TFOCS/blob/master/linop_matrix.m]]
 */
class LinopMatrix(private val matrix: DMatrix) extends LinearOperator[DenseVector, DVector] {

  if (matrix.getStorageLevel == StorageLevel.NONE) {
    matrix.cache()
  }

  override def apply(x: DenseVector): DVector = {
    val bcX = matrix.context.broadcast(x)
    // Take the dot product of each matrix row with x.
    // NOTE A DenseVector result is assumed here (not sparse safe).
    matrix.mapPartitions(partitionRows =>
      Iterator.single(new DenseVector(partitionRows.map(row => BLAS.dot(row, bcX.value)).toArray)))
  }

  override def t: LinearOperator[DVector, DenseVector] = new LinopMatrixAdjoint(matrix)
}
