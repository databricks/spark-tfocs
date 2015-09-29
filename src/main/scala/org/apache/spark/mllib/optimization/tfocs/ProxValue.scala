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

/**
 * The evaluation result of a ProxCapableFunction. A ProxCapableFunction may be evaluated to find
 * the prox minimizer and/or the function value at the prox minimizer. A ProxValue encapsulates
 * these evaluation results.
 *
 * @param f The function value at the prox minimizer, if computed.
 * @param minimizer The prox minimizer, if computed.
 * @tparam X A type representing a vector.
 *
 * @see [[org.apache.spark.mllib.optimization.tfocs.ProxCapableFunction]]
 */
case class ProxValue[X](f: Option[Double], minimizer: Option[X])
