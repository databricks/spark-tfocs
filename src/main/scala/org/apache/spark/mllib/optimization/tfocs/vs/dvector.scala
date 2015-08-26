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

package org.apache.spark.mllib.optimization.tfocs.vs

import org.apache.spark.mllib.linalg.BLAS
import org.apache.spark.mllib.optimization.tfocs.VectorSpace
import org.apache.spark.mllib.optimization.tfocs.VectorSpace._
import org.apache.spark.mllib.optimization.tfocs.vs.vector.DenseVectorSpace
import org.apache.spark.storage.StorageLevel

package object dvector {

  /** A VectorSpace for DVector vectors. */
  implicit object DVectorSpace extends VectorSpace[DVector] {

    import org.apache.spark.mllib.optimization.tfocs.DVectorFunctions._

    override def combine(alpha: Double, a: DVector, beta: Double, b: DVector): DVector =
      a.zip(b).map(_ match {
        case (aPart, bPart) =>
          // NOTE A DenseVector result is assumed here (not sparse safe).
          DenseVectorSpace.combine(alpha, aPart, beta, bPart).toDense
      })

    override def dot(a: DVector, b: DVector): Double =
      a.zip(b).aggregate(0.0)((sum, x) => sum + BLAS.dot(x._1, x._2), _ + _)

    override def cache(a: DVector): Unit =
      if (a.getStorageLevel == StorageLevel.NONE) {
        a.cache()
      }
  }
}
