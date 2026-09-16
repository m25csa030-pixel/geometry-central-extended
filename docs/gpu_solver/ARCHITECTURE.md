# Architecture Analysis: Linear Solvers in Geometry-Central

This document analyzes the existing sparse linear solver architecture in `geometry-central`, explaining the CPU solver flow, matrix storage representations, interfaces, memory ownership patterns, and integration points for a GPU-accelerated backend.

---

## 1. Current CPU Solver Flow

`geometry-central` separates linear systems into two distinct lifecycle phases:
1. **Factorization / Setup Phase**: Expensive symbolic and numeric prefactorization executed once in the constructor.
2. **Solve Phase**: Lightweight forward and backward triangular substitutions executed across one or multiple right-hand sides.

```
+-------------------------------------------------------------------------------+
|                             FACTORIZATION PHASE                               |
|                                                                               |
|   Eigen::SparseMatrix<T>                                                      |
|           │                                                                   |
|           ▼                                                                   |
|   PositiveDefiniteSolver<T>::PositiveDefiniteSolver(SparseMatrix<T>& mat)      |
|           │                                                                   |
|           ├──> checkFinite(mat); checkHermitian(mat); mat.makeCompressed();   |
|           │                                                                   |
|   #ifdef GC_HAVE_SUITESPARSE                     #else (Eigen Fallback)       |
|           │                                                │                  |
|           ▼                                                ▼                  |
|   toCholmod(mat, context, SType::SYMMETRIC)      internals->solver.compute()  |
|           │                                                │                  |
|   cholmod_l_analyze(...)                                   │ (SimplicialLDLT) |
|   cholmod_l_factorize(...)                                 │                  |
|           │                                                │                  |
|           ▼                                                ▼                  |
|   internals->factorization (LDL^T)                 internals->solver          |
+-------------------------------------------------------------------------------+
                                    │
                                    ▼
+-------------------------------------------------------------------------------+
|                                 SOLVE PHASE                                   |
|                                                                               |
|   Vector<T> rhs                                                               |
|           │                                                                   |
|           ▼                                                                   |
|   PositiveDefiniteSolver<T>::solve(Vector<T>& x, const Vector<T>& rhs)        |
|           │                                                                   |
|   #ifdef GC_HAVE_SUITESPARSE                     #else (Eigen Fallback)       |
|           │                                                │                  |
|           ▼                                                ▼                  |
|   cholmod_dense* inVec = toCholmod(rhs)          x = internals->solver.       |
|   cholmod_dense* outVec = cholmod_l_solve(..)         solve(rhs)              |
|   toEigen(outVec, context, x)                              │                  |
|   cholmod_l_free_dense(...)                                │                  |
|           │                                                │                  |
|           ▼                                                ▼                  |
|       Output Vector<T> x                               Output Vector<T> x     |
+-------------------------------------------------------------------------------+
```

### 1.1. SuiteSparse CHOLMOD Backend
When `GC_HAVE_SUITESPARSE` is defined at compile time:
* `toCholmod()` transforms the input `Eigen::SparseMatrix<double>` into a `cholmod_sparse` object.
* `context.setSimplicial()` and `context.setLDL()` enforce simplicial $L D L^T$ factorization (avoiding expensive fill-in computation for typical 2D manifold mesh connectivity).
* `cholmod_l_analyze()` analyzes the non-zero sparsity pattern and generates an elimination tree.
* `cholmod_l_factorize()` calculates the numerical values of the factors.
* In `solve()`: the RHS vector is copied into `cholmod_dense`, solved via `cholmod_l_solve(CHOLMOD_A, ...)`, and copied back to Eigen `Vector<T>`.

### 1.2. Eigen SimplicialLDLT Fallback
When SuiteSparse is not present:
* `internals->solver` is an instance of `Eigen::SimplicialLDLT<SparseMatrix<T>>`.
* `solver.compute(mat)` performs direct $L D L^T$ decomposition.
* `solver.solve(rhs)` evaluates the solution directly on the CPU.

---

## 2. Sparse Matrix Representation

### 2.1. Type Hierarchy & Invariants
In `include/geometrycentral/numerical/linear_algebra_types.h`:
```cpp
template <typename T>
using SparseMatrix = Eigen::SparseMatrix<T>; // Defaults to Eigen::ColMajor
```

Eigen's `SparseMatrix<T, Eigen::ColMajor>` uses Compressed Sparse Column (CSC) format:
* `outerIndexPtr()`: array of length $N_{\text{cols}} + 1$, holding starting offsets of each column.
* `innerIndexPtr()`: array of length $NNZ$ (number of non-zeros), holding 0-based row indices.
* `valuePtr()`: array of length $NNZ$, holding floating-point coefficients.

### 2.2. Structural Symmetry: CSC vs. CSR
For any symmetric matrix $A = A^T$:
$$\text{CSC}(A) \equiv \text{CSR}(A^T) = \text{CSR}(A)$$
Consequently:
* The column pointers (`outerIndexPtr()`) of the ColMajor Eigen matrix correspond **identically** to CSR row pointers (`csrRowOffsets`).
* The row indices (`innerIndexPtr()`) correspond **identically** to CSR column indices (`csrColInd`).
* The values array (`valuePtr()`) corresponds to CSR non-zero values (`csrVal`).

This structural equivalence enables zero-copy or direct `cudaMemcpy` from Eigen sparse memory to GPU CSR structures without requiring an expensive matrix transposition step on the CPU.

### 2.3. Construction Pattern
Sparse matrices throughout `geometry-central` (such as `cotanLaplacian`, `vertexLumpedMassMatrix`, and `vertexConnectionLaplacian`) are constructed using the triplet pattern:
```cpp
std::vector<Eigen::Triplet<T>> triplets;
// Loop over halfedges / vertices / faces...
triplets.emplace_back(row, col, value);
// Assembly:
SparseMatrix<T> mat(nRows, nCols);
mat.setFromTriplets(triplets.begin(), triplets.end());
mat.makeCompressed();
```

---

## 3. Solver Interface & Abstraction

In `include/geometrycentral/numerical/linear_solvers.h`:

```cpp
template <typename T>
class LinearSolver {
public:
  LinearSolver(const SparseMatrix<T>& mat) : nRows(mat.rows()), nCols(mat.cols()) {}
  virtual ~LinearSolver() {}

  virtual Vector<T> solve(const Vector<T>& rhs) = 0;
  virtual void solve(Vector<T>& x, const Vector<T>& rhs) = 0;

protected:
  size_t nRows, nCols;
};
```

### Derived Classes:
* `PositiveDefiniteSolver<T>`: Implements `LinearSolver<T>` with private implementation `std::unique_ptr<PSDSolverInternals<T>> internals;`.
* `SquareSolver<T>`: Implements general square solves using `SquareSolverInternals<T>`.
* `Solver<T>`: Implements QR solves for general rectangular or rank-deficient systems.

### Non-Member Helpers:
```cpp
template <typename T>
Vector<T> solvePositiveDefinite(SparseMatrix<T>& matrix, const Vector<T>& rhs);
```
Constructs a temporary `PositiveDefiniteSolver<T>` and executes a single solve.

---

## 4. Data Ownership and Lifecycle

### 4.1. Input Matrix Ownership
* The input `SparseMatrix<T>& mat` is passed by reference to `LinearSolver` constructors.
* The solver does **not** take ownership of the caller's matrix.
* In SuiteSparse mode, `toCholmod()` copies the entries into `cholmod_sparse`.
* In Eigen mode, `internals->solver.compute(mat)` copies and stores factor matrices internally.

### 4.2. Vector Ownership
* Right-hand side `rhs` is passed as `const Vector<T>&`.
* Solutions are either returned by value (`Vector<T>`) or populated into an existing vector `Vector<T>& x`.
* No vector buffers are retained between invocations on the CPU.

### 4.3. Algorithmic Lifecycles (Heat Method & Vector Heat Method)
In `HeatMethodDistanceSolver`:
```cpp
std::unique_ptr<PositiveDefiniteSolver<double>> heatSolver;
std::unique_ptr<PositiveDefiniteSolver<double>> poissonSolver;
```
* **Persistent Factorization**: Solvers are instantiated in the constructor and stored as members.
* `computeDistance()` can be called repeatedly (e.g., from different source vertices or source points) without re-factoring the matrix:
  * Only Step 1 (`heatSolver->solve()`) and Step 3 (`poissonSolver->solve()`) are re-evaluated per query.
* In `VectorHeatMethodSolver`:
  * Solvers (`scalarHeatSolver`, `vectorHeatSolver`, `poissonSolver`, `affineHeatSolver`) are lazily initialized via `ensureHave...()` methods and retained for subsequent vector transport and logarithmic map queries.

---

## 5. Expected Integration Points for GPU Acceleration

```
+---------------------------------------------------------------------------------+
|                                 INTEGRATION ARCHITECTURE                        |
|                                                                                 |
|                        LinearSolver<T> (Abstract Base)                          |
|                                       ▲                                         |
|                                       │                                         |
|                 ┌─────────────────────┴─────────────────────┐                   |
|                 │                                           │                   |
|     PositiveDefiniteSolver<T>                   [NEW] CUDAPositiveDefinite-     |
|     (Existing CPU Direct LDLT)                        Solver<T> / PCG           |
|                 │                                           │                   |
|         ┌───────┴───────┐                                   ├── CuSPARSE SpMV   |
|         ▼               ▼                                   ├── CuBLAS (DOT,    |
|    SuiteSparse        Eigen                                 │           AXPY)   |
|     (CHOLMOD)    (SimplicialLDLT)                           └── Jacobi Precond  |
|                                                                                 |
+---------------------------------------------------------------------------------+
```

### Integration Point A: `LinearSolver<T>` Polymorphism
* Because `HeatMethodDistanceSolver` and `VectorHeatMethodSolver` interact with solvers through the `solve(rhs)` interface, introducing a GPU backend can be achieved cleanly either by:
  1. Adding a backend selector within `PositiveDefiniteSolver<T>` (redirecting to GPU internals when requested), OR
  2. Providing a `CUDAPCGPositiveDefiniteSolver<T>` that implements `LinearSolver<T>`, allowing algorithmic classes to accept solver instances via `std::unique_ptr<LinearSolver<T>>`.

### Integration Point B: Host-to-Device Matrix Transfer
* The Eigen `SparseMatrix<double>` can be uploaded to GPU memory directly:
  * `d_csrRowOffsets` $\leftarrow$ `mat.outerIndexPtr()` (size $N+1$)
  * `d_csrColInd` $\leftarrow$ `mat.innerIndexPtr()` (size $NNZ$)
  * `d_csrVal` $\leftarrow$ `mat.valuePtr()` (size $NNZ$)
* The diagonal entries $A_{ii}$ are readily extracted to compute the inverse diagonal Jacobi preconditioner $M^{-1} = \text{diag}(1 / A_{ii})$ on the GPU.

### Integration Point C: Algorithmic Pipeline Integration
In `HeatMethodDistanceSolver::computeDistanceRHS`:
1. `heatSolver->solve(rhsVec)`:
   * CPU: Host vector $\rightarrow$ GPU $\rightarrow$ PCG solve $\rightarrow$ GPU result.
2. Gradient & Divergence evaluation:
   * Immediate baseline: Device-to-Host transfer $\rightarrow$ CPU face gradient and divergence loop $\rightarrow$ Host-to-Device transfer.
   * Optimized pipeline: Keep data in GPU memory and evaluate gradient & divergence via CUDA kernels, eliminating round-trip PCIe transfers entirely.
3. `poissonSolver->solve(divergenceVec)`:
   * PCG solve on regularized Laplacian operator ($L + 10^{-6} I$).

