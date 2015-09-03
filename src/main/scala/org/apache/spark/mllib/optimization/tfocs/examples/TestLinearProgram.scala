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

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.tfocs.DVectorFunctions._
import org.apache.spark.mllib.optimization.tfocs.SolverSLP
import org.apache.spark.mllib.optimization.tfocs.fs.dvector.vector.LinopMatrix
import org.apache.spark.mllib.random.{ RandomDataGenerator, RandomRDDs }
import org.apache.spark.mllib.rdd.RandomVectorRDD
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.util.random.XORShiftRandom

/**
 * Helper class for generating sparse standard normal values.
 *
 * @param density The density of non sparse values.
 */
private class SparseStandardNormalGenerator(density: Double) extends RandomDataGenerator[Double] {

  private val random = new XORShiftRandom()

  override def nextValue(): Double = if (random.nextDouble < density) random.nextGaussian else 0.0

  override def setSeed(seed: Long): Unit = random.setSeed(seed)

  override def copy(): SparseStandardNormalGenerator = new SparseStandardNormalGenerator(density)
}

/**
 * This example generates a random linear programming problem and solves it using SolverSLP.
 *
 * The example can be executed as follows:
 * sbt 'test:run-main org.apache.spark.mllib.optimization.tfocs.examples.TestLinearProgram'
 *
 * NOTE In matlab tfocs this example can be found in test_LinearProgram.m.
 * @see [[https://github.com/cvxr/TFOCS/blob/master/examples/smallscale/test_LinearProgram.m]]
 */
object TestLinearProgram {
  def main(args: Array[String]) {

    val rnd = new Random(34324)
    val sparkConf = new SparkConf().setMaster("local[2]").setAppName("TestLinearProgram")
    val sc = new SparkContext(sparkConf)

    val n = 5000 // Tranpose constraint matrix row count.
    val m = n / 2 // Transpose constrint matrix column count.

    // Generate a starting 'x' vector, using normally generated values.
    val x = RandomRDDs.normalRDD(sc, n).map(_ + 10).glom.map(Vectors.dense(_).toDense)

    // Generate the transpose constraint matrix 'A' using sparse normally generated values.
    val A = new RandomVectorRDD(sc,
      n,
      m,
      sc.defaultMinPartitions,
      new SparseStandardNormalGenerator(0.01),
      rnd.nextLong)

    // Generate the cost vector 'c' using normally generated values.
    val c = RandomRDDs.normalRDD(sc, n, 0, rnd.nextLong).glom.map(Vectors.dense(_).toDense)

    // Compute 'b' using the starting 'x' vector.
    val b = new LinopMatrix(A)(x)

    val mu = 1e-2
    val x0 = sc.parallelize(new Array[Double](n)).glom.map(Vectors.dense(_).toDense)
    val z0 = Vectors.zeros(m).toDense

    // Solve the linear program using SolverSLP, finding the optimal x vector 'optimalX'.
    val (optimalX, _) = SolverSLP.run(c, A, b, mu, x0, z0)
    println("optimalX: " + optimalX.collectElements.mkString(", "))

    sc.stop()
  }
}
