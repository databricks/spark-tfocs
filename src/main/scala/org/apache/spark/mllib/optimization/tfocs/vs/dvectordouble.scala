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

import org.apache.spark.mllib.optimization.tfocs.VectorSpace
import org.apache.spark.mllib.optimization.tfocs.VectorSpace._
import org.apache.spark.mllib.optimization.tfocs.vs.dvector.DVectorSpace

package object dvectordouble {

  /*
   * A VectorSpace for (DVector, Double) pairs. Such pairs arise when two separate functions are
   * applied to an input value.
   */
  implicit object DVectorDoubleSpace extends VectorSpace[(DVector, Double)] {

    override def combine(alpha: Double,
      a: (DVector, Double),
      beta: Double,
      b: (DVector, Double)): (DVector, Double) =
      (DVectorSpace.combine(alpha, a._1, beta, b._1), alpha * a._2 + beta * b._2)

    override def dot(a: (DVector, Double), b: (DVector, Double)): Double =
      DVectorSpace.dot(a._1, b._1) + a._2 * b._2

    override def cache(a: (DVector, Double)): Unit = DVectorSpace.cache(a._1)
  }
}
