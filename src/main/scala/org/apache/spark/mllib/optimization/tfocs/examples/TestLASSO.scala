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

package org.apache.spark.mllib.optimization.tfocs.examples

import scala.util.Random

import org.apache.spark.mllib.linalg.{ BLAS, DenseVector, Vectors }
import org.apache.spark.mllib.optimization.tfocs.SolverL1RLS
import org.apache.spark.mllib.random.RandomRDDs
import org.apache.spark.{ SparkConf, SparkContext }

/**
 * This example generates a random lasso optimization problem and solves it using SolverL1RLS.
 *
 * The example can be executed as follows:
 * sbt 'test:run-main org.apache.spark.mllib.optimization.tfocs.examples.TestLASSO'
 *
 * NOTE In matlab tfocs this functionality can be found in test_LASSO.m.
 * @see [[https://github.com/cvxr/TFOCS/blob/master/examples/smallscale/test_LASSO.m]]
 */
object TestLASSO {
  def main(args: Array[String]) {

    val rnd = new Random(34324)
    val sparkConf = new SparkConf().setMaster("local[2]").setAppName("TestLASSO")
    val sc = new SparkContext(sparkConf)

    val n = 1024 // Design matrix column count.
    val m = n / 2 // Design matrix row count.
    val k = m / 5 // Count of nonzero weights.

    // Generate the design matrix using random normal values, then normalize the columns.
    val unnormalizedA = RandomRDDs.normalVectorRDD(sc, m, n, 0, rnd.nextLong)
    val AColumnNormSq = unnormalizedA.treeAggregate(Vectors.zeros(n).toDense)(
      seqOp = (sum, rowA) => {
        val rowASq = Vectors.dense(rowA.toArray.map(rowA_i => rowA_i * rowA_i))
        BLAS.axpy(1.0, rowASq, sum)
        sum
      },
      combOp = (sum1, sum2) => {
        BLAS.axpy(1.0, sum2, sum1)
        sum1
      })
    val A = unnormalizedA.map(rowA =>
      Vectors.dense(rowA.toArray.zip(AColumnNormSq.toArray).map {
        case (rowA_i, normsq_i) => rowA_i / math.sqrt(normsq_i)
      }))

    // Generate the actual 'x' vector, including 'k' nonzero values.
    val x = Vectors.zeros(n).toDense
    for (i <- rnd.shuffle(1 to n).take(k)) {
      x.values(i) = rnd.nextGaussian
    }

    // Generate the 'b' vector using the design matrix and weights, adding gaussian noise.
    val bOriginal = new DenseVector(A.map(rowA => BLAS.dot(rowA, x)).collect)
    val snr = 30 // SNR in dB
    val sigma =
      math.pow(10, ((10 * math.log10(math.pow(Vectors.norm(bOriginal, 2), 2) / n) - snr) / 20))
    val b = sc.parallelize(bOriginal.values.map(_ + sigma * rnd.nextGaussian))
      .glom
      .map(new DenseVector(_))

    // Set 'lambda' using the noise standard deviation.
    val lambda = 2 * sigma * math.sqrt(2 * math.log(n))

    // Solve the lasso problem using SolverL1RLS, finding the estimated x vector 'estimatedX'.
    val x0 = Vectors.zeros(n).toDense
    val (estimatedX, _) = SolverL1RLS.run(A, b, lambda, x0)
    println("estimatedX: " + estimatedX.values.mkString(", "))

    sc.stop()
  }
}
