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

import org.scalatest.{ FunSuite, Matchers }

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.mllib.util.TestingUtils._

class TFOCSSuite extends FunSuite with MLlibTestSparkContext with Matchers {

  test("The weights and losses returned by Spark TFOCS should match those returned by Matlab " +
    "TFOCS") {

    // The test below checks that the results match those of the following TFOCS matlab program
    // (using TFOCS version 1945a771f315acd4cc6eba638b5c01fb52ee7aaa):
    //
    // A = [ -0.8307    0.2722    0.1947   -0.3545    0.3944   -0.5557   -0.2904    0.5337   -0.1190    0.0657;
    //        0.2209   -0.2547   -0.4508   -0.1773   -0.0596   -0.2363    0.1157   -0.2136    0.4888   -0.2178;
    //        0.2045   -0.2948   -0.1195   -0.3493   -0.6571    0.0966    0.4472   -0.0572   -0.0996   -0.7121;
    //        0.4056   -0.6623    0.7984    0.3474    0.0084   -0.0191    0.5596   -0.4359    0.8581    0.0490;
    //       -0.2341   -0.5792    0.3272   -0.7748    0.6396   -0.7910   -0.6239   -0.6901    0.0249    0.6624; ];
    // b = [ 0.1614; -0.1662; 0.4224; -0.2945; -0.3866 ];
    // lambda = 0.0298;
    // x0 = zeros(10, 1);
    // [x, out] = solver_L1RLS(A, b, lambda, x0, struct('restart', -Inf));
    // format long
    // x
    // out.f

    val A = sc.parallelize(Array(
      Vectors.dense(-0.8307, 0.2722, 0.1947, -0.3545, 0.3944, -0.5557, -0.2904, 0.5337, -0.1190,
        0.0657),
      Vectors.dense(0.2209, -0.2547, -0.4508, -0.1773, -0.0596, -0.2363, 0.1157, -0.2136, 0.4888,
        -0.2178),
      Vectors.dense(0.2045, -0.2948, -0.1195, -0.3493, -0.6571, 0.0966, 0.4472, -0.0572, -0.0996,
        -0.7121),
      Vectors.dense(0.4056, -0.6623, 0.7984, 0.3474, 0.0084, -0.0191, 0.5596, -0.4359, 0.8581,
        0.0490),
      Vectors.dense(-0.2341, -0.5792, 0.3272, -0.7748, 0.6396, -0.7910, -0.6239, -0.6901, 0.0249,
        0.6624)), 1)
    val b = sc.parallelize(Array(Vectors.dense(0.1614, -0.1662, 0.4224, -0.2945, -0.3866)), 1)
    val lambda = 0.0298
    val x0 = Vectors.zeros(10)

    val (x, lossHistory) = TFOCS.optimize(new SmoothQuadRDDVector(b),
      new ProductVectorRDDVector(A),
      new ProxL1Vector(lambda),
      x0)

    val expectedX = Vectors.dense(-0.049755786974910, 0, 0.076369527414210, 0, 0, 0, 0,
      0.111550837996771, -0.314626347477663, -0.503782689620966)
    val expectedLossHistory = Array(0.113425611210499, 0.077669187887145, 0.061961458212103,
      0.052553214376800, 0.046562416223286, 0.042826602488959, 0.040906876606451, 0.040100239630275,
      0.039547309449369, 0.039102810446688, 0.038743759101449, 0.038473157445446, 0.038243690177961,
      0.037820176724358, 0.037674646255497, 0.037595407614782, 0.037541273717515, 0.037488658690212,
      0.037461351107222, 0.037436427062658, 0.037402935660960, 0.037359705908762, 0.037305269198822,
      0.037237950582842, 0.037155899829656, 0.037057087628944, 0.037008220702622, 0.036953926171892,
      0.036893600152694, 0.036826574489721, 0.036752122970015, 0.036669591881800, 0.036580127509047,
      0.036514112601292, 0.036466676943336, 0.036437875734202, 0.036421012039977, 0.036403014246581,
      0.036379560291601, 0.036350464595941, 0.036315702958934, 0.036275255783379, 0.036228988068812,
      0.036176533601874, 0.036117172611430, 0.036049746292837, 0.035972652609517, 0.035883935137560,
      0.035799830060464, 0.035762477906535, 0.035728652612072, 0.035699143752643, 0.035673550241229,
      0.035651117275342, 0.035631078694737, 0.035612837652109, 0.035578885955199, 0.035547574511839,
      0.035517810586924, 0.035497171087318, 0.035482808673008, 0.035467004342919, 0.035456070725224,
      0.035453198535721, 0.035451417707606, 0.035450121160635, 0.035449346728343, 0.035448999442808,
      0.035448931753041, 0.035448902405041, 0.035448899798522, 0.035448909338006, 0.035448919645683,
      0.035448923916734, 0.035448920668238, 0.035448914628844, 0.035448909763521, 0.035448905992926,
      0.035448903196490, 0.035448901223834, 0.035448899909616, 0.035448899089566, 0.035448898614999,
      0.035448898363268, 0.035448898242601, 0.035448898191233, 0.035448898172230, 0.035448898166288,
      0.035448898164774, 0.035448898164507, 0.035448898164472, 0.035448898164460, 0.035448898164453,
      0.035448898164448, 0.035448898164443, 0.035448898164439, 0.035448898164435, 0.035448898164432,
      0.035448898164430, 0.035448898164428, 0.035448898164427, 0.035448898164426, 0.035448898164425,
      0.035448898164425, 0.035448898164425, 0.035448898164425, 0.035448898164425)

    assert(x ~= expectedX relTol 1e-6,
      "Each weight vector element should match the expected value, within tolerance.")

    assert(Vectors.dense(lossHistory) ~= Vectors.dense(expectedLossHistory) relTol 1e-12,
      "The loss value on each iteration should match the expected value, within tolerance.")
  }
}
