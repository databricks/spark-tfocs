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

import java.io.File

import com.joptimizer.optimizers.LPStandardConverter
import com.joptimizer.util.MPSParser

import org.apache.spark.mllib.linalg.{ DenseVector, Vector, Vectors }
import org.apache.spark.mllib.optimization.tfocs.DVectorFunctions._
import org.apache.spark.mllib.optimization.tfocs.SolverSLP
import org.apache.spark.{ SparkConf, SparkContext }

/**
 * This example reads a linear program in MPS format and solves it using SolverSLP.
 *
 * The example can be executed as follows:
 * sbt 'test:run-main
 *   org.apache.spark.mllib.optimization.tfocs.examples.TestLinearProgram <mps file>'
 */
object TestMPSLinearProgram {
  def main(args: Array[String]) {

    val sparkConf = new SparkConf().setMaster("local[2]").setAppName("TestMPSLinearProgram")
    val sc = new SparkContext(sparkConf)

    // Parse the provided MPS file.
    val parser = new MPSParser()
    var mpsFile = new File(args(0))
    parser.parse(mpsFile)

    // Convert the parsed linear program to standard form.
    val converter = new LPStandardConverter(true)
    converter.toStandardForm(parser.getC,
      parser.getG,
      parser.getH,
      parser.getA,
      parser.getB,
      parser.getLb,
      parser.getUb)

    // Convert the parameters of the linear program to the proper formats.
    val c = sc.parallelize(converter.getStandardC.toArray).glom.map(new DenseVector(_))
    val A = sc.parallelize(converter.getStandardA.toArray.transpose.map(
      Vectors.dense(_).toSparse: Vector))
    val b = new DenseVector(converter.getStandardB.toArray)
    val n = converter.getStandardN

    val mu = 1e-2
    val x0 = sc.parallelize(new Array[Double](n)).glom.map(new DenseVector(_))
    val z0 = Vectors.zeros(b.size).toDense

    // Solve the linear program using SolverSLP, finding the optimal x vector 'optimalX'.
    val (optimalX, _) = SolverSLP.run(c, A, b, mu, x0, z0)
    println("optimalX: " + optimalX.collectElements.mkString(", "))

    sc.stop()
  }
}
