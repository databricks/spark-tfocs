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

import org.apache.spark.mllib.linalg.{ BLAS, DenseVector, Vectors }
import org.apache.spark.mllib.optimization.tfocs.VectorSpace._

/**
 * Extra functions available on DVectors through an implicit conversion. DVectors are represented
 * using RDD[DenseVector], and these helper functions apply operations to the values within each
 * DenseVector of the RDD.
 */
private[tfocs] class DVectorFunctions(self: DVector) {

  def mapElements(f: Double => Double): DVector = self.map(x => new DenseVector(x.values.map(f)))

  def zipElements(other: DVector, f: (Double, Double) => Double): DVector =
    self.zip(other).map(_ match {
      case (selfPart, otherPart) =>
        if (selfPart.size != otherPart.size) {
          throw new IllegalArgumentException("Can only zipElements DVectors with the same number " +
            "of elements")
        }
        // NOTE DenseVectors are assumed here (not sparse safe).
        val ret = new Array[Double](selfPart.size)
        var i = 0
        while (i < ret.size) {
          ret(i) = f(selfPart(i), otherPart(i))
          i += 1
        }
        new DenseVector(ret)
    })

  def aggregateElements(zeroValue: Double)(
    seqOp: (Double, Double) => Double,
    combOp: (Double, Double) => Double): Double =
    self.aggregate(zeroValue)(
      seqOp = (aggregate, vector) => {
        // NOTE DenseVectors are assumed here (not sparse safe).
        val vectorAggregate = vector.values.aggregate(zeroValue)(seqop = seqOp, combop = combOp)
        combOp(vectorAggregate, aggregate)
      },
      combOp = combOp)

  def collectElements: Array[Double] =
    // NOTE DenseVectors are assumed here (not sparse safe).
    self.collect().flatMap(_.values)

  def diff(other: DVector): DVector =
    self.zip(other).map(_ match {
      case (selfPart, otherPart) =>
        val ret = selfPart.copy
        BLAS.axpy(-1.0, otherPart, ret)
        ret
    })

  def sqdist(other: DVector): Double =
    self.zip(other).aggregate(0.0)((sum, x) => sum + Vectors.sqdist(x._1, x._2), _ + _)

  def sum: Double = self.aggregate(0.0)((sum, x) => sum + x.values.sum, _ + _)

  def dot(other: DVector): Double =
    self.zip(other).aggregate(0.0)((sum, x) => sum + BLAS.dot(x._1, x._2), _ + _)
}

private[tfocs] object DVectorFunctions {

  implicit def DVectorToDVectorFunctions(dVector: DVector): DVectorFunctions =
    new DVectorFunctions(dVector)

  def axpy(a: Double, x: DVector, y: DVector) =
    x.zip(y).map(_ match {
      case (xPart, yPart) =>
        val ret = yPart.copy
        BLAS.axpy(a, xPart, ret)
        ret
    })
}
