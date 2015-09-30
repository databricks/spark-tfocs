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
* Multiple data distribution patterns. (Support is currently only implemented for RDD[Vector] row
  matrices.)

## Examples

Solve the l1 regularized least squares problem `0.5 * ||A * x' - b||_2^2 + lambda * ||x||_1` (lasso
linear regression):

    import org.apache.spark.mllib.linalg.{ DenseVector, Vectors }
    import org.apache.spark.mllib.optimization.tfocs.SolverL1RLS
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
        0.6624)), 2)
    val b = sc.parallelize(Array(0.1614, -0.1662, 0.4224, -0.2945, -0.3866), 2).glom.map(
      new DenseVector(_))
    val lambda = 0.0298

    SolverL1RLS.run(A, b, lambda)

Alternatively, the above optimization may be executed using the TFOCS optimizer directly rather than
with the `SolverL1RLS` helper implementation:

    import org.apache.spark.mllib.optimization.tfocs.fs.dvector.double._
    import org.apache.spark.mllib.optimization.tfocs.fs.vector.double._
    import org.apache.spark.mllib.optimization.tfocs.fs.vector.dvector._
    import org.apache.spark.mllib.optimization.tfocs.TFOCS
    import org.apache.spark.mllib.optimization.tfocs.vs.dvector._
    import org.apache.spark.mllib.optimization.tfocs.vs.vector._
    val x0 = Vectors.zeros(10).toDense

    TFOCS.optimize(new SmoothQuad(b), new LinopMatrix(A), new ProxL1(lambda), x0)

To solve the smoothed standard form linear program:

    minimize c' * x + 0.5 * mu * ||x - x0||_2^2
    s.t.     A' * x == b' and x >= 0

<!-- code block break -->

    import org.apache.spark.mllib.linalg.{ DenseVector, Vectors }
    import org.apache.spark.mllib.optimization.tfocs.SolverSLP
    val A = sc.parallelize(Array(Vectors.zeros(5),
      Vectors.sparse(5, Seq((1, 0.632374636716572), (4, 0.198436985375040))),
      Vectors.sparse(5, Seq((2, 0.179885783103202))), Vectors.zeros(5), Vectors.zeros(5),
      Vectors.zeros(5), Vectors.zeros(5), Vectors.sparse(5, Seq((1, 0.014792694748719))),
      Vectors.zeros(5), Vectors.sparse(5, Seq((3, 0.244326895623829)))), 2)
    var b = new DenseVector(Array(0, 7.127414296861894, 1.781441255102280, 2.497425876822379,
      2.186136752456199))
    val c = sc.parallelize(Array(-1.078275146772097, -0.368208440839284, 0.680376092886272,
      0.256371934668609, 1.691983132986665, 0.059837119884475, -0.221648385883038,
      -0.298134575377277, -1.913199010346937, 0.745084172661387), 2).glom.map(
      new DenseVector(_))
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
  dot products. This abstraction supports local computation and also distributed computation using
  implementations based on different data distribution models.

* `LinearOperator` An interface for performing a linear mapping from one vector space to another.

* `SmoothFunction` An interface for evaluating a smooth function and computing its gradient.

* `ProxCapableFunction` An interface for evaluating a function and computing the minimizing value
  of its proximity operator.

The following naming conventions are used in the Spark TFOCS library:

* To the extent possible, classes and functions are given the same names as the corresponding
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
