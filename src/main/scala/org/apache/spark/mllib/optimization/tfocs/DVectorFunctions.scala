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

  /** Apply a function to each DVector element. */
  def mapElements(f: Double => Double): DVector =
    self.map(part => new DenseVector(part.values.map(f)))

  /**
   * Zip a DVector's elements with those of another DVector and apply a function to each pair of
   * elements.
   */
  def zipElements(other: DVector, f: (Double, Double) => Double): DVector =
    self.zip(other).map {
      case (selfPart, otherPart) =>
        if (selfPart.size != otherPart.size) {
          throw new IllegalArgumentException("Can only call zipElements on DVectors with the " +
            "same number of elements and consistent partitions.")
        }
        // NOTE DenseVectors are assumed here (not sparse safe).
        val ret = new Array[Double](selfPart.size)
        var i = 0
        while (i < ret.size) {
          ret(i) = f(selfPart(i), otherPart(i))
          i += 1
        }
        new DenseVector(ret)
    }

  /** Apply aggregation functions to the DVector elements. */
  def aggregateElements(zeroValue: Double)(
    seqOp: (Double, Double) => Double,
    combOp: (Double, Double) => Double): Double =
    self.aggregate(zeroValue)(
      seqOp = (aggregate, part) => {
        // NOTE DenseVectors are assumed here (not sparse safe).
        val partAggregate = part.values.aggregate(zeroValue)(seqop = seqOp, combop = combOp)
        combOp(partAggregate, aggregate)
      },
      combOp = combOp)

  /** Collect the DVector elements to a local array. */
  def collectElements: Array[Double] =
    // NOTE DenseVectors are assumed here (not sparse safe).
    self.collect().flatMap(_.values)

  /** Compute the elementwise difference of this DVector with another. */
  def diff(other: DVector): DVector =
    self.zip(other).map {
      case (selfPart, otherPart) =>
        val ret = selfPart.copy
        BLAS.axpy(-1.0, otherPart, ret)
        ret
    }

  /** Sum the DVector's elements. */
  def sum: Double = self.aggregate(0.0)((sum, x) => sum + x.values.sum, _ + _)

  /** Compute the dot product with another DVector. */
  def dot(other: DVector): Double =
    self.zip(other).aggregate(0.0)((sum, x) => sum + BLAS.dot(x._1, x._2), _ + _)
}

private[tfocs] object DVectorFunctions {

  implicit def DVectorToDVectorFunctions(dVector: DVector): DVectorFunctions =
    new DVectorFunctions(dVector)
}
