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

import org.apache.spark.mllib.linalg.{ BLAS, DenseVector, Vector }
import org.apache.spark.rdd.RDD

class VectorRDDFunctions(self: RDD[Vector]) {

  def mapElements(f: Double => Double): RDD[Vector] =
    self.map(x => new DenseVector(x.toArray.map(f)))

  def zipElements(other: RDD[Vector], f: (Double, Double) => Double): RDD[Vector] =
    self.zip(other).map({ x =>
      if (x._1.size != x._2.size) {
        throw new IllegalArgumentException("Can only zipElements RDD[Vector]s with the same " +
          "number of elements")
      }
      new DenseVector(x._1.toArray.zip(x._2.toArray).map(y => f(y._1, y._2))): Vector
    })

  def diff(other: RDD[Vector]): RDD[Vector] =
    self.zip(other).map({ x =>
      val ret = x._1.copy
      BLAS.axpy(-1.0, x._2, ret)
      ret
    })

  def sum: Double = self.map(_.toArray.sum).sum
}

object VectorRDDFunctions {

  implicit def RDDToVectorRDDFunctions(rdd: RDD[Vector]): VectorRDDFunctions =
    new VectorRDDFunctions(rdd)
}
