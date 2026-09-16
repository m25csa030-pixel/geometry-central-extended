# File Map: GPU-Accelerated Solver Integration for Geometry-Central

This document maps all repository files relevant to the GPU-accelerated sparse SPD linear solver, Jacobi-preconditioned Conjugate Gradient (PCG) backend, and downstream heat-method workflows.

---

## 1. Numerical Solver Interfaces & Implementations

### `include/geometrycentral/numerical/linear_solvers.h`
* **Purpose**: Core abstract interface and concrete class declarations for linear solvers across geometry-central.
* **Important Classes / Functions**:
  * `template <typename T> class LinearSolver`: Abstract base class defining virtual methods `virtual Vector<T> solve(const Vector<T>& rhs) = 0` and `virtual void solve(Vector<T>& x, const Vector<T>& rhs) = 0`. Holds matrix dimensions `nRows`, `nCols`.
  * `template <typename T> class PositiveDefiniteSolver final : public LinearSolver<T>`: Concrete solver for symmetric/Hermitian positive definite systems using PIMPL idiom (`std::unique_ptr<PSDSolverInternals<T>> internals`).
  * `template <typename T> class SquareSolver final : public LinearSolver<T>`: Concrete solver for general square nonsymmetric/indefinite systems.
  * `template <typename T> class Solver final : public LinearSolver<T>`: QR-based least-squares / minimum-norm solver.
  * `template <typename T> Vector<T> solvePositiveDefinite(SparseMatrix<T>& matrix, const Vector<T>& rhs)`: Non-member one-shot solver helper.
  * `template <typename T> double residual(const SparseMatrix<T>& matrix, const Vector<T>& lhs, const Vector<T>& rhs)`: Computes $\|A x - b\|_2$.
* **Dependencies**:
  * `geometrycentral/numerical/linear_algebra_utilities.h`
  * `<Eigen/Sparse>`
  * `<memory>`, `<iostream>`

### `src/numerical/linear_solvers.cpp`
* **Purpose**: Explicit template instantiations of `LinearSolver<T>` and implementation of the non-member `residual()` helper.
* **Important Classes / Functions**:
  * Explicit instantiations: `LinearSolver<float>`, `LinearSolver<double>`, `LinearSolver<std::complex<double>>`.
  * `residual<T>(matrix, lhs, rhs)`: Evaluates $\|A x - b\|_2$ using conjugate transpose.
* **Dependencies**:
  * `geometrycentral/numerical/linear_solvers.h`
  * `geometrycentral/numerical/linear_algebra_utilities.h`
  * `geometrycentral/utilities/vector2.h`

### `src/numerical/positive_definite_solvers.cpp`
* **Purpose**: CPU implementation of `PositiveDefiniteSolver<T>` for symmetric/Hermitian positive definite systems.
* **Important Classes / Functions**:
  * `template <typename T> struct PSDSolverInternals`: Private implementation struct containing:
    * Under `GC_HAVE_SUITESPARSE`: `CholmodContext context; cholmod_sparse* cMat; cholmod_factor* factorization;`
    * Fallback: `Eigen::SimplicialLDLT<SparseMatrix<T>> solver;`
  * `PositiveDefiniteSolver<T>::PositiveDefiniteSolver(SparseMatrix<T>& mat)`: Constructor validating symmetry (`checkHermitian`), finiteness (`checkFinite`), compressing matrix (`makeCompressed`), and performing symbolic + numeric Cholesky/LDLt factorization.
  * `PositiveDefiniteSolver<T>::solve(Vector<T>& x, const Vector<T>& rhs)`: Forward/backward substitution solve.
  * Explicit template instantiations for `float`, `double`, `std::complex<double>`.
* **Dependencies**:
  * `geometrycentral/numerical/linear_solvers.h`
  * `geometrycentral/numerical/linear_algebra_utilities.h`
  * `geometrycentral/numerical/suitesparse_utilities.h` (under `#ifdef GC_HAVE_SUITESPARSE`)

### `src/numerical/square_solvers.cpp`
* **Purpose**: CPU implementation of `SquareSolver<T>` for general square systems using UMFPACK or Eigen `SparseLU`.
* **Important Classes / Functions**:
  * `SquareSolverInternals<T>`: Holds UMFPACK symbolic and numeric factorization handles or `Eigen::SparseLU`.
  * Used by `VectorHeatMethodSolver` when meshes fail the Delaunay condition or for affine heat flow.
* **Dependencies**:
  * `geometrycentral/numerical/linear_solvers.h`
  * `geometrycentral/numerical/suitesparse_utilities.h`
  * `<umfpack.h>`

---

## 2. Linear Algebra Core Types & Utilities

### `include/geometrycentral/numerical/linear_algebra_types.h`
* **Purpose**: Fundamental typedefs establishing Eigen as the linear algebra backing library.
* **Important Classes / Typedefs**:
  * `template <typename T> using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>`
  * `template <typename T> using SparseMatrix = Eigen::SparseMatrix<T>` (ColMajor default)
  * `template <typename T> using DenseMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>`
  * `EigenVectorMap_T`, `ConstEigenVectorMap_T`
* **Dependencies**:
  * `<Eigen/Core>`
  * `<Eigen/Sparse>`

### `include/geometrycentral/numerical/linear_algebra_utilities.h` & `linear_algebra_utilities.ipp`
* **Purpose**: Utility functions for matrix creation, manipulation, validation, and serialization.
* **Important Classes / Functions**:
  * `identityMatrix<T>(size_t N)`: Creates sparse identity matrix.
  * `shiftDiagonal<T>(SparseMatrix<T>& m, T shiftAmount)`: Adds diagonal offset $\lambda I$.
  * `checkFinite()`, `checkSymmetric()`, `checkHermitian()`: Matrix sanity checkers.
  * `blockDecomposeSquare()`: Interleaved 2x2 block decomposition.
  * `complexToReal()`, `realToComplex()`: Isomorphic mapping between complex $N \times N$ and real $2N \times 2N$ linear systems.
* **Dependencies**:
  * `geometrycentral/numerical/linear_algebra_types.h`
  * `geometrycentral/utilities/utilities.h`
  * `<Eigen/Dense>`, `<Eigen/Sparse>`

### `include/geometrycentral/numerical/suitesparse_utilities.h` & `src/numerical/suitesparse_utilities.cpp`
* **Purpose**: Interop wrappers between Eigen matrices/vectors and SuiteSparse CHOLMOD structures.
* **Important Classes / Functions**:
  * `class CholmodContext`: RAII wrapper around `cholmod_common`.
  * `toCholmod(Eigen::SparseMatrix<T>& A, ...)`: Converts Eigen CSC format (`valuePtr()`, `innerIndexPtr()`, `outerIndexPtr()`) into `cholmod_sparse`.
  * `toCholmod(Eigen::Matrix<T>& v, ...)`: Converts Eigen dense column vector into `cholmod_dense`.
  * `toEigen(cholmod_dense*, ...)`: Copies CHOLMOD dense solution back into Eigen `Vector<T>`.
* **Dependencies**:
  * `<cholmod.h>`, `<SuiteSparseQR.hpp>` (conditional on `GC_HAVE_SUITESPARSE`)
  * `geometrycentral/numerical/linear_algebra_utilities.h`

---

## 3. Surface Algorithms (Heat Method & Vector Heat Method)

### `include/geometrycentral/surface/heat_method_distance.h`
* **Purpose**: Class interface for computing geodesic distance via the Heat Method (Crane, Weischedel, Wardetzky 2013, 2017).
* **Important Classes / Functions**:
  * `class HeatMethodDistanceSolver`: Stateful solver precomputing operators for repeated geodesic queries.
  * `VertexData<double> computeDistance(const Vertex& sourceVert)`: Computes geodesic distance field.
  * `Vector<double> computeDistanceRHS(const Vector<double>& rhs)`: Evaluates heat flow, vector gradient/divergence, and Poisson solve.
  * Members:
    * `std::unique_ptr<PositiveDefiniteSolver<double>> heatSolver`: Solves $(M + t L) u = \delta$.
    * `std::unique_ptr<PositiveDefiniteSolver<double>> poissonSolver`: Solves $(L + \epsilon I) \phi = \text{div}(X)$.
* **Dependencies**:
  * `geometrycentral/numerical/linear_solvers.h`
  * `geometrycentral/surface/intrinsic_geometry_interface.h`
  * `geometrycentral/surface/surface_mesh.h`

### `src/surface/heat_method_distance.cpp`
* **Purpose**: Implementation of `HeatMethodDistanceSolver`.
* **Important Steps / Functions**:
  * Constructor:
    * Calculates short diffusion time: $t = t_{\text{coef}} \cdot \bar{h}^2$.
    * Assembles heat operator matrix $A_{\text{heat}} = M + t L$.
    * Instantiates `heatSolver = std::make_unique<PositiveDefiniteSolver<double>>(heatOp)`.
    * Assembles regularized Poisson operator $A_{\text{poisson}} = L + 10^{-6} I$.
    * Instantiates `poissonSolver = std::make_unique<PositiveDefiniteSolver<double>>(Ls)`.
  * `computeDistanceRHS(rhsVec)`:
    * Step 1: Solve heat diffusion: `heatVec = heatSolver->solve(rhsVec)`.
    * Step 2: Compute face gradients $\nabla u$, normalize to unit vector field $X = -\nabla u / \|\nabla u\|$, evaluate vertex divergence $\nabla \cdot X$.
    * Step 3: Solve Poisson equation: `distVec = poissonSolver->solve(divergenceVec)`.
  * Robust mode: supports `useRobustLaplacian = true` (tufted cover and intrinsic Delaunay mollification via `mollifyIntrinsic` and `flipToDelaunay`).
* **Dependencies**:
  * `geometrycentral/surface/heat_method_distance.h`
  * `geometrycentral/surface/tufted_laplacian.h`
  * `geometrycentral/surface/simple_idt.h`
  * `geometrycentral/surface/intrinsic_mollification.h`

### `include/geometrycentral/surface/vector_heat_method.h`
* **Purpose**: Interface for Vector Heat Method (Sharp, Soliman, Crane 2019) computing parallel transport and logarithmic maps.
* **Important Classes / Functions**:
  * `class VectorHeatMethodSolver`: Stateful solver precomputing scalar and vector connection heat flows.
  * Members:
    * `std::unique_ptr<PositiveDefiniteSolver<double>> scalarHeatSolver`: Scalar heat diffusion $(M + t L)$.
    * `std::unique_ptr<LinearSolver<std::complex<double>>> vectorHeatSolver`: Vector connection heat diffusion $(M + t L_{\text{conn}})$.
    * `std::unique_ptr<PositiveDefiniteSolver<double>> poissonSolver`: Distance integration $(L)$.
    * `std::unique_ptr<LinearSolver<double>> affineHeatSolver`: 3x3 affine heat flow.
* **Dependencies**:
  * `geometrycentral/numerical/linear_solvers.h`
  * `geometrycentral/surface/heat_method_distance.h`
  * `geometrycentral/surface/intrinsic_geometry_interface.h`

### `src/surface/vector_heat_method.cpp`
* **Purpose**: Implementation of `VectorHeatMethodSolver`.
* **Important Steps / Functions**:
  * `ensureHaveVectorHeatSolver()`:
    * Assembles connection heat operator $M_{\mathbb{C}} + t L_{\text{conn}}$.
    * If mesh is Delaunay (all cotan weights non-negative), attempts `PositiveDefiniteSolver<std::complex<double>>`; falls back to `SquareSolver<std::complex<double>>`.
  * `transportTangentVectors()`, `computeLogMap()`: Invokes `vectorHeatSolver->solve()`, `scalarHeatSolver->solve()`, `poissonSolver->solve()`.
* **Dependencies**:
  * `geometrycentral/surface/vector_heat_method.h`
  * `geometrycentral/numerical/linear_algebra_utilities.h`

---

## 4. Geometry & Sparse Matrix Assembly

### `include/geometrycentral/surface/intrinsic_geometry_interface.h` & `src/surface/intrinsic_geometry_interface.cpp`
* **Purpose**: Central cached property manager for mesh geometric quantities and differential operators.
* **Important Members / Functions**:
  * `SparseMatrix<double> cotanLaplacian`: Standard cotangent Laplacian operator.
  * `SparseMatrix<double> vertexLumpedMassMatrix`: Diagonal lumped dual vertex areas.
  * `SparseMatrix<std::complex<double>> vertexConnectionLaplacian`: Parallel transport connection Laplacian.
  * Assembly pattern: Iterates over edges/halfedges, accumulates into `std::vector<Eigen::Triplet<T>>`, and invokes `setFromTriplets()`.
* **Dependencies**:
  * `geometrycentral/surface/base_geometry_interface.h`
  * `geometrycentral/numerical/linear_algebra_types.h`

### `include/geometrycentral/surface/tufted_laplacian.h` & `src/surface/tufted_laplacian.cpp`
* **Purpose**: Generates intrinsic Delaunay tufted covers for non-manifold and degenerate CAD geometries.
* **Important Functions**:
  * `buildIntrinsicTuftedCover()`
  * `buildTuftedLaplacian()`
* **Relevance**: Crucial for testing and evaluating robustness on degenerate CAD meshes (Objective 6).

---

## 5. Build System & CMake Configuration

### `CMakeLists.txt` (Root)
* **Purpose**: Top-level CMake project configuration (`geometry-central`).
* **Key Configuration**:
  * Sets C++ standard and package export definitions.
  * Subdirectories: `deps`, `src`.
  * Propagates `GC_HAVE_SUITESPARSE`.

### `deps/CMakeLists.txt`
* **Purpose**: Dependency resolution for Eigen3, SuiteSparse (CHOLMOD/UMFPACK), and vendored libraries (`nanort`, `nanoflann`, `happly`).
* **Key Configuration**:
  * `find_package(Eigen3 3.3 QUIET)` or auto-download.
  * `option(SUITESPARSE "Enable SuiteSparse." ON)`: finds SuiteSparse/CHOLMOD/UMFPACK, sets `GC_HAVE_SUITESPARSE`.

### `src/CMakeLists.txt`
* **Purpose**: Compiles the `geometry-central` library target.
* **Key Configuration**:
  * Source list `SRCS` containing all numerical, surface, pointcloud, and utility source files.
  * Public includes, compile feature `cxx_std_11`.
  * Conditional inclusion of SuiteSparse headers and libraries.

### `test/CMakeLists.txt`
* **Purpose**: Builds test executable `geometry-central-test`.
* **Key Configuration**:
  * Uses `FetchContent` to download GoogleTest (v1.15.2).
  * Defines assets path `-DGC_TEST_ASSETS_ABS_PATH="..."`.
  * Links `gtest_main` and `geometry-central`.

---

## 6. Testing, Benchmarking & Utilities

### `include/geometrycentral/utilities/timing.h`
* **Purpose**: High-precision wall-clock timing macros and formatting.
* **Important Macros**:
  * `START_TIMING(name)`: Captures `std::chrono::steady_clock::now()`.
  * `FINISH_TIMING(name)`: Returns elapsed microseconds.
  * `FINISH_TIMING_SEC(name)`: Returns elapsed seconds as `double`.
  * `FINISH_TIMING_PRINT(name)`: Prints human-readable elapsed time.

### `test/src/linear_algebra_test.cpp`
* **Purpose**: Unit tests for matrix operations and linear solvers.
* **Important Test Cases**:
  * `LinearAlgebraTestSuite::buildSPDTestMatrix<T>()`: Generates strictly SPD test matrices from mesh graph Laplacian with random positive weights.
  * `TEST_F(LinearAlgebraTestSuite, TestLDLTSolvers)`: Tests `PositiveDefiniteSolver` across `float`, `double`, and `std::complex<double>`, asserting residual $< 10^{-4}$.
  * `TEST_F(LinearAlgebraTestSuite, TestSquareSolvers)`: Tests `SquareSolver`.

### `test/include/load_test_meshes.h` & `test/src/load_test_meshes.cpp`
* **Purpose**: Helper harness to load sample meshes from `test/assets/` (`spot.ply`, `bob_small.ply`, `lego.ply`, `cat_head.obj`, etc.).

