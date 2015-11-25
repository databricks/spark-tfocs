# spark-tfocs

Spark TFOCS is an implementation of the [TFOCS](http://cvxr.com/tfocs/) convex solver for [Apache
Spark](http://spark.apache.org/).

The original Matlab TFOCS library provides building blocks to construct efficient solvers for convex
problems. Spark TFOCS implements a useful subset of this functionality, in Scala, and is designed to
operate on distributed data using the Spark cluster computing framework. Spark TFOCS includes
support for:

* Convex optimization using Nesterov's accelerated method (Auslender and Teboulle variant)
* Adaptive step size using backtracking Lipschitz estimation
* Automatic acceleration restart using the gradient test
* Linear operator structure optimizations
* Smoothed Conic Dual (SCD) formulation solver, with continuation support
* Smoothed linear program solver
* Multiple data distribution patterns. (Currently support is only implemented for RDD[Vector] row
  matrices.)

The name "TFOCS" is being used by the original TFOCS developers, who are not involved in the
development of this package and hence not responsible for the support.
To report issues or request features about Spark TFOCS, please use our GitHub issues page.


## LASSO Example

Solve the l1 regularized least squares problem `0.5 * ||A * x' - b||_2^2 + lambda * ||x||_1` (lasso
linear regression):

    import org.apache.spark.mllib.linalg.{ DenseVector, Vectors }
    import org.apache.spark.mllib.optimization.tfocs.SolverL1RLS

    // Design matrix
    val A = sc.parallelize(Array(
      Vectors.dense(0.61, 0.98, 0.32),
      Vectors.dense(0.10, 0.22, 0.92),
      Vectors.dense(0.79, 0.02, 0.20)), 2)

    // Observations
    val b = sc.parallelize(Array(3.69, 3.36, 1.59), 2).glom.map(new DenseVector(_))

    // Regularization term
    val lambda = 0.1

    SolverL1RLS.run(A, b, lambda)

Alternatively, the above optimization may be performed using the TFOCS optimizer directly rather
than via the `SolverL1RLS` helper:

    import org.apache.spark.mllib.optimization.tfocs.fs.dvector.double._
    import org.apache.spark.mllib.optimization.tfocs.fs.vector.double._
    import org.apache.spark.mllib.optimization.tfocs.fs.vector.dvector._
    import org.apache.spark.mllib.optimization.tfocs.TFOCS
    import org.apache.spark.mllib.optimization.tfocs.vs.dvector._
    import org.apache.spark.mllib.optimization.tfocs.vs.vector._

    // Initial x vector
    val x0 = Vectors.zeros(3).toDense

    TFOCS.optimize(new SmoothQuad(b), new LinopMatrix(A), new ProxL1(lambda), x0)

## Linear Program Example

To solve the smoothed standard form linear program:

    minimize c' * x + 0.5 * mu * ||x - x0||_2^2
    s.t.     A' * x == b' and x >= 0

<!-- code block break -->

    import org.apache.spark.mllib.linalg.{ DenseVector, Vectors }
    import org.apache.spark.mllib.optimization.tfocs.SolverSLP

    // Constraint matrix
    val A = sc.parallelize(Array(
      Vectors.sparse(3, Seq((0, 0.88))),
      Vectors.sparse(3, Seq((1, 0.63))),
      Vectors.sparse(3, Seq((0, 0.29), (2, 0.18)))), 2)

    // Constraint vector
    var b = new DenseVector(Array(9.50, 6.84, 5.09))

    // Objective vector
    val c = sc.parallelize(Array(1.0, 2.0, 3.0), 2).glom.map(new DenseVector(_))

    // Smoothing parameter
    val mu = 1e-2

    SolverSLP.run(c, A, b, mu)

## Solvers

* `SolverL1RLS` A solver for lasso problems.
* `SolverSLP` A solver for smoothed standard form linear programs.
* `TFOCS` A general purpose convex solver.
* `TFOCS_SCD` A solver for problems using the TFOCS Smooth Conic Dual formulation.

## Software Architecture Overview

The primary types used in the Spark TFOCS library are as follows:

* `DenseVector` A wrapper around `Array[Double]` with support for vector operations. (Imported
  from `org.apache.spark.mllib.linalg`)

* `DVector` A distributed vector, stored as an `RDD[DenseVector]`, where each partition comprises a
  single `DenseVector` containing a slice of the complete distributed vector. More information is
  available in `org.apache.spark.mllib.optimization.tfocs.VectorSpace`.

* `DMatrix` A distributed matrix, stored as an `RDD[Vector]`, where each (possibly sparse) `Vector`
  represents a row of the matrix. More information is available in
  `org.apache.spark.mllib.optimization.tfocs.VectorSpace`.

The primary abstractions of the Spark TFOCS library are as follows:

* `VectorSpace` A basic vector space interface with support for computing linear combinations and
  dot products. This abstraction supports local computation as well as distributed computation using
  implementations based on different data distribution models.

* `LinearOperator` An interface for performing a linear mapping from one vector space to another.

* `SmoothFunction` An interface for evaluating a smooth function and computing its gradient.

* `ProxCapableFunction` An interface for evaluating a function and computing the minimizing value
  of its proximity operator.

The following naming conventions are used in the Spark TFOCS library:

* To the extent possible, classes and functions are given the same name as the corresponding
  implementation in Matlab TFOCS.

* `VectorSpace` implementations are placed in the `vs` namespace. For example the `VectorSpace` for
  `DVector` vectors is named `vs.dvector`.

* Function implementations (implementations of `LinearOperator`, `SmoothFunction`, and
  `ProxCapableFunction`) are placed in the `fs` (function space) namespace, and are specifically
  named according to their input and output types. For example, functions with input type `DVector`
  and output type `Double` are placed in the `fs.dvector.double` namespace.

## TODOs

* Block matrix cluster distribution pattern.
* Block matrix sparse storage format.
* Efficient computation on sparse vectors (not just sparse matrices).
* Arbitrary vector space support in TFOCS_SCD.
* Additional objective functions.
