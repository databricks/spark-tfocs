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

package org.apache.spark.mllib.optimization.tfocs.fs.dvector.double

import org.apache.spark.mllib.optimization.tfocs.DVectorFunctions._
import org.apache.spark.mllib.optimization.tfocs.{ Mode, SmoothFunction, Value }
import org.apache.spark.mllib.optimization.tfocs.VectorSpace._
import org.apache.spark.storage.StorageLevel

/**
 * The log likelihood logistic loss function applied to a DVector.
 *
 * @param y The observed values, labeled as binary 0/1.
 * @param mu The variable values.
 *
 * NOTE In matlab tfocs this functionality is implemented in smooth_logLLogistic.m.
 * @see [[https://github.com/cvxr/TFOCS/blob/master/smooth_logLLogistic.m]]
 */
class SmoothLogLLogistic(y: DVector) extends SmoothFunction[DVector] with Serializable {

  if (y.getStorageLevel == StorageLevel.NONE) {
    y.cache()
  }

  override def apply(mu: DVector, mode: Mode): Value[DVector] = {

    val f = if (mode.f) {
      // TODO Performance might be improved by reimplementing as a single aggregate rather than
      // mapping through an intermediate DVector and summing, which breaks per-element pipelining.
      Some(y.zipElements(mu, (y_i, mu_i) => {
        val yFactor = if (mu_i > 0.0) y_i - 1.0 else if (mu_i < 0.0) y_i else 0.0
        yFactor * mu_i - math.log1p(math.exp(-math.abs(mu_i)))
      }).sum)
    } else {
      None
    }

    val g = if (mode.g) {
      Some(y.zipElements(mu, (y_i, mu_i) => {
        // Compute the log logistic loss gradient elementwise.
        val muFactor = if (mu_i > 0.0) 1.0 else math.exp(mu_i)
        y_i - muFactor / (1.0 + math.exp(-math.abs(mu_i)))
      }))
    } else {
      None
    }

    Value(f, g)
  }
}
