# Sparse SPD Solver Architecture Analysis

This document provides a comprehensive technical analysis of the sparse Symmetric Positive-Definite (SPD) solver architecture in `geometry-central`, addressing the 10 core architectural and integration questions.

---

## 1. What is the exact solver interface?

The base solver interface is defined in [`include/geometrycentral/numerical/linear_solvers.h`](../../include/geometrycentral/numerical/linear_solvers.h) (lines 58–73):

```cpp
namespace geometrycentral {

template <typename T>
class LinearSolver {
public:
  LinearSolver(const SparseMatrix<T>& mat) : nRows(mat.rows()), nCols(mat.cols()) {}
  virtual ~LinearSolver() {}

  // Solve for a particular right hand side
  virtual Vector<T> solve(const Vector<T>& rhs) = 0;

  // Solve for a particular right hand side, and return in an existing vector object
  virtual void solve(Vector<T>& x, const Vector<T>& rhs) = 0;

protected:
  size_t nRows, nCols;
};

} // namespace geometrycentral
```

### Supporting Non-Member Functions
In [`include/geometrycentral/numerical/linear_solvers.h`](../../include/geometrycentral/numerical/linear_solvers.h) (lines 51, 55):
* `template <typename T> Vector<T> solvePositiveDefinite(SparseMatrix<T>& matrix, const Vector<T>& rhs);`  
  Constructs a temporary `PositiveDefiniteSolver<T>` and returns the solution for one-off solves.
* `template <typename T> double residual(const SparseMatrix<T>& matrix, const Vector<T>& lhs, const Vector<T>& rhs);`  
  Implemented in [`src/numerical/linear_solvers.cpp`](../../src/numerical/linear_solvers.cpp) (lines 14–19); evaluates $\|A x - b\|_2$.

---

## 2. Which classes implement it?

All concrete solvers derive from `LinearSolver<T>` and use the PIMPL pattern to encapsulate backend details:

1. **`PositiveDefiniteSolver<T>`** (SPD direct solver):
   * Declaration: [`include/geometrycentral/numerical/linear_solvers.h`](../../include/geometrycentral/numerical/linear_solvers.h) (lines 102–114).
   * Implementation: [`src/numerical/positive_definite_solvers.cpp`](../../src/numerical/positive_definite_solvers.cpp).
   * Backends: SuiteSparse CHOLMOD (simplicial $L D L^T$) or `Eigen::SimplicialLDLT<SparseMatrix<T>>`.
2. **`SquareSolver<T>`** (General square solver for non-symmetric/indefinite systems):
   * Declaration: [`include/geometrycentral/numerical/linear_solvers.h`](../../include/geometrycentral/numerical/linear_solvers.h) (lines 119–132).
   * Implementation: [`src/numerical/square_solvers.cpp`](../../src/numerical/square_solvers.cpp).
   * Backends: SuiteSparse UMFPACK or `Eigen::SparseLU<SparseMatrix<T>>`.
3. **`Solver<T>`** (QR solver for rectangular/least-squares/minimum-norm systems):
   * Declaration: [`include/geometrycentral/numerical/linear_solvers.h`](../../include/geometrycentral/numerical/linear_solvers.h) (lines 81–97).
   * Implementation: [`src/numerical/qr_solvers.cpp`](../../src/numerical/qr_solvers.cpp).
   * Backends: SuiteSparse SPQR or `Eigen::SPQR<SparseMatrix<T>>` / dense QR fallback.

### Explicit Template Instantiations
Explicitly instantiated in [`src/numerical/linear_solvers.cpp`](../../src/numerical/linear_solvers.cpp) (lines 9–11) and [`src/numerical/positive_definite_solvers.cpp`](../../src/numerical/positive_definite_solvers.cpp) (lines 142–144):
* `LinearSolver<float>`, `LinearSolver<double>`, `LinearSolver<std::complex<double>>`
* `PositiveDefiniteSolver<float>`, `PositiveDefiniteSolver<double>`, `PositiveDefiniteSolver<std::complex<double>>`

---

## 3. How are sparse matrices represented?

In [`include/geometrycentral/numerical/linear_algebra_types.h`](../../include/geometrycentral/numerical/linear_algebra_types.h) (lines 15–16):
```cpp
template <typename T>
using SparseMatrix = Eigen::SparseMatrix<T>;
```

### Layout & Storage:
* By default, `Eigen::SparseMatrix<T>` uses **Column-Major** ordering, corresponding to **Compressed Sparse Column (CSC)** format.
* The internal structure is accessed via three contiguous buffers:
  1. `mat.valuePtr()`: Array of non-zero numerical values (`T*`) of size `mat.nonZeros()`.
  2. `mat.innerIndexPtr()`: Array of row indices (`StorageIndex*`, typically `int`) of size `mat.nonZeros()`.
  3. `mat.outerIndexPtr()`: Array of column starting offsets (`StorageIndex*`) of size `mat.cols() + 1`.

### CSC-to-CSR Structural Equivalence for Symmetric Matrices:
For any symmetric matrix $A = A^T$:
$$\text{CSC}(A) \equiv \text{CSR}(A^T) = \text{CSR}(A)$$
* Column starting offsets (`outerIndexPtr()`) match **CSR row offsets** (`csrRowOffsets`).
* Inner row indices (`innerIndexPtr()`) match **CSR column indices** (`csrColInd`).
* Values (`valuePtr()`) match **CSR values** (`csrVal`).
* **Benefit**: No CPU transposition or reordering is needed; Eigen CSC data pointers can be copied directly into GPU CSR buffers (`cudaMemcpy`).

### Assembly Pattern:
Geometric operators (e.g., `cotanLaplacian`, `vertexLumpedMassMatrix`) are built using `Eigen::Triplet<T>` lists:
```cpp
std::vector<Eigen::Triplet<double>> triplets;
// Populate triplets...
SparseMatrix<double> mat(nRows, nCols);
mat.setFromTriplets(triplets.begin(), triplets.end());
mat.makeCompressed();
```

---

## 4. How are RHS vectors and solution vectors represented?

In [`include/geometrycentral/numerical/linear_algebra_types.h`](../../include/geometrycentral/numerical/linear_algebra_types.h) (lines 11–12):
```cpp
template <typename T>
using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
```

### Representation & Properties:
* Standard dense contiguous 1D column vector.
* Pointer to raw data: `v.data()` (`T*`).
* Dimension: `v.rows()` elements.
* Direct transfer to GPU: `cudaMemcpy(d_v, v.data(), sizeof(T) * v.rows(), ...)`.

### Interop with Mesh Geometry Containers:
Mesh quantities (e.g., `VertexData<T>`) interoperate seamlessly with `Vector<T>`:
* Export to vector: `Vector<double> v = vertexData.toVector();`
* Import from vector: `VertexData<double> vertexData(mesh, v);`

---

## 5. Is the matrix assembled once and solved multiple times?

**Yes.** `geometry-central` algorithms strictly separate operator construction from repeated solves to amortize expensive matrix setup.

### Evidence in `HeatMethodDistanceSolver`:
In [`include/geometrycentral/surface/heat_method_distance.h`](../../include/geometrycentral/surface/heat_method_distance.h) (lines 70–71) and [`src/surface/heat_method_distance.cpp`](../../src/surface/heat_method_distance.cpp) (lines 60–69, 188, 210):
* Matrices $A_{\text{heat}} = M + t L$ and $A_{\text{poisson}} = L + 10^{-6} I$ are assembled and factored **once** in the constructor:
  ```cpp
  heatSolver.reset(new PositiveDefiniteSolver<double>(heatOp));
  poissonSolver.reset(new PositiveDefiniteSolver<double>(Ls));
  ```
* In `computeDistance(sourceVert)` or `computeDistanceRHS(rhsVec)`, the factored solvers are reused:
  ```cpp
  Vector<double> heatVec = heatSolver->solve(rhsVec);
  // ... compute face gradients and divergence ...
  Vector<double> distVec = poissonSolver->solve(divergenceVec);
  ```
* The solver instances persist for the entire lifetime of `HeatMethodDistanceSolver`, serving any number of distance queries.

### Evidence in `VectorHeatMethodSolver`:
In [`include/geometrycentral/surface/vector_heat_method.h`](../../include/geometrycentral/surface/vector_heat_method.h) (lines 72–75) and [`src/surface/vector_heat_method.cpp`](../../src/surface/vector_heat_method.cpp) (lines 31–83):
* Solvers (`scalarHeatSolver`, `vectorHeatSolver`, `poissonSolver`) are instantiated once on demand via `ensureHave...()` methods and reused across all scalar extension, parallel transport, and logarithmic map evaluations.

---

## 6. How are factorization and solve separated?

### 1. Factorization / Setup Phase
Executed in `PositiveDefiniteSolver<T>::PositiveDefiniteSolver(SparseMatrix<T>& mat)` ([`src/numerical/positive_definite_solvers.cpp`](../../src/numerical/positive_definite_solvers.cpp), lines 36–84):
* **Validation**:
  * Asserts `nRows == nCols`.
  * `#ifndef GC_NLINALG_DEBUG`: runs `checkFinite(mat)` and `checkHermitian(mat)`.
  * Enforces `mat.makeCompressed()`.
* **Execution**:
  * **SuiteSparse (`GC_HAVE_SUITESPARSE`)**:
    * `toCholmod(mat, context, SType::SYMMETRIC)` copies matrix to `cholmod_sparse`.
    * `context.setSimplicial()` and `context.setLDL()` select simplicial $L D L^T$.
    * `cholmod_l_analyze(internals->cMat, internals->context)` constructs symbolic factorization and elimination tree.
    * `cholmod_l_factorize(internals->cMat, internals->factorization, internals->context)` computes numerical values.
  * **Eigen Fallback**:
    * `internals->solver.compute(mat)` performs direct $L D L^T$ decomposition.
* **Storage**: Factors are stored inside `internals` (`PSDSolverInternals<T>`).

### 2. Solve Phase
Executed in `PositiveDefiniteSolver<T>::solve(Vector<T>& x, const Vector<T>& rhs)` ([`src/numerical/positive_definite_solvers.cpp`](../../src/numerical/positive_definite_solvers.cpp), lines 94–132):
* **Validation**: Asserts `rhs.rows() == N` and `checkFinite(rhs)`.
* **Execution**:
  * **SuiteSparse**: Wraps `rhs` via `toCholmod`, executes forward/backward substitution via `cholmod_l_solve(CHOLMOD_A, internals->factorization, inVec, ...)`, copies back via `toEigen`, and frees vector wrappers.
  * **Eigen Fallback**: `x = internals->solver.solve(rhs)`.
* **Cost**: $O(NNZ)$ triangular substitution; zero matrix refactorization.

---

## 7. Which methods must a CUDA backend implement?

A complete CUDA backend must provide an implementation conforming to `LinearSolver<T>`:

```cpp
template <typename T>
class CUDAPCGPositiveDefiniteSolver : public LinearSolver<T> {
public:
  // 1. Setup / Factorization Phase
  CUDAPCGPositiveDefiniteSolver(SparseMatrix<T>& mat, double tol = 1e-6, size_t maxIters = 1000);
  ~CUDAPCGPositiveDefiniteSolver();

  // 2. Pure Virtual Solve Overrides
  void solve(Vector<T>& x, const Vector<T>& rhs) override;
  Vector<T> solve(const Vector<T>& rhs) override;

  // 3. Solver Parameter Tuning & Diagnostics
  void setTolerance(double tol);
  void setMaxIterations(size_t maxIters);
  size_t getIterationsAchieved() const;
  double getFinalResidual() const;
};
```

### Detailed Lifecycle Operations:
* **Constructor**:
  * Upload CSR row offsets, column indices, and values to device memory (`d_rowOffsets`, `d_colIndices`, `d_values`).
  * Extract diagonal elements $A_{ii}$ and build inverted Jacobi preconditioner on device: $d_{\text{inv}}[i] = 1.0 / A_{ii}$.
  * Initialize cuSPARSE matrix descriptor (`cusparseCreateCsr`) and query SpMV workspace buffer size.
  * Pre-allocate working vectors on device: $x, r, z, p, q$ (size $N$).
* **`solve(x, rhs)`**:
  * Copy `rhs` from host to device `d_b`.
  * Initialize $x_0 = 0$, $r_0 = b$, $z_0 = M^{-1} r_0$, $p_0 = z_0$, $\gamma_0 = r_0^T z_0$.
  * Execute PCG loop:
    * `cusparseSpMV` for $q = A p$.
    * `cublasDdot` for $p^T q$.
    * `cublasDaxpy` for $x$ and $r$ updates.
    * Diagonal preconditioner kernel: $z_i = r_i \cdot d_{\text{inv}}[i]$.
    * Check residual norm $\|r\|_2 < \text{tol} \cdot \|b\|_2$.
  * Copy device solution `d_x` back to host `x`.
* **Destructor**:
  * Free device memory (`cudaFree`) and destroy cuSPARSE descriptors and cuBLAS handles.

---

## 8. How do heat-method algorithms use this interface?

In [`src/surface/heat_method_distance.cpp`](../../src/surface/heat_method_distance.cpp):

### 1. Setup in Constructor (lines 52–74):
```cpp
// Heat operator: (M + t * L)
SparseMatrix<double> heatOp = M + shortTime * L;
heatSolver.reset(new PositiveDefiniteSolver<double>(heatOp));

// Poisson operator: (L + 1e-6 * I)
SparseMatrix<double> Ls = L + 1e-6 * identityMatrix<double>(mesh.nVertices());
poissonSolver.reset(new PositiveDefiniteSolver<double>(Ls));
```

### 2. Execution in `computeDistanceRHS` (lines 180–219):
```cpp
// Step 1: Solve heat diffusion
Vector<double> heatVec = heatSolver->solve(rhsVec);

// Step 2: CPU geometry processing
// Loops over faces, computes face gradients gradUDir, normalizes,
// and evaluates vertex divergence into divergenceVec.

// Step 3: Solve Poisson equation
Vector<double> distVec = poissonSolver->solve(divergenceVec);
```

### Architectural Observation:
In [`include/geometrycentral/surface/heat_method_distance.h`](../../include/geometrycentral/surface/heat_method_distance.h) (lines 70–71):
```cpp
std::unique_ptr<PositiveDefiniteSolver<double>> heatSolver;
std::unique_ptr<PositiveDefiniteSolver<double>> poissonSolver;
```
Members are declared as `std::unique_ptr<PositiveDefiniteSolver<double>>` rather than `std::unique_ptr<LinearSolver<double>>`. Upgrading these member types to `LinearSolver<double>` will allow transparent substitution of any CPU or GPU solver backend.

---

## 9. What compatibility constraints exist with Eigen?

1. **Header Compilation Isolation**:
   * CUDA device compiler (`nvcc`) can produce warnings or incompatibilities with certain Eigen header templates.
   * **Rule**: Keep `.cu` files isolated to pure CUDA/cuSPARSE/cuBLAS C APIs. Pass raw pointers (`T*`, `int*`) across translation units, rather than including heavy Eigen headers inside CUDA device compilation units.
2. **Memory Layout**:
   * `Vector<T>` is dense and contiguous; `v.data()` maps directly to `cudaMemcpy`.
   * `SparseMatrix<T>` is CSC. For symmetric matrices, CSC maps directly to CSR without data permuting.
3. **Index Types**:
   * `Eigen::SparseMatrix::StorageIndex` defaults to signed 32-bit `int`.
   * cuSPARSE requires specifying `CUSPARSE_INDEX_32I` to match Eigen's `int` index buffers.
4. **Exception Handling**:
   * Eigen returns status codes (`solver.info() != Eigen::Success`).
   * `geometry-central` converts failures into standard C++ exceptions (`std::runtime_error`, `std::invalid_argument`).
   * The CUDA backend must check CUDA/cuSPARSE/cuBLAS return codes and throw descriptive `std::runtime_error` exceptions on failure.

---

## 10. Which design choices are required for GPU integration?

### Choice 1: Class Hierarchy & Polymorphism
* **Recommended Strategy**: Implement `CUDAPCGPositiveDefiniteSolver<T>` derived from `LinearSolver<T>`.
* In `HeatMethodDistanceSolver` and `VectorHeatMethodSolver`, type solver pointers as `std::unique_ptr<LinearSolver<double>>`.
* Provide an optional constructor parameter `SolverBackend backend = SolverBackend::Default` (or `CPU` vs `CUDA_PCG`).

### Choice 2: Zero-Allocation `solve()` Pattern
* All device working buffers ($x, r, z, p, q$, $d_{\text{inv}}$, SpMV workspace) must be allocated **once** during `CUDAPCGPositiveDefiniteSolver` construction and persisted.
* `solve()` must perform **zero** device memory allocations, ensuring maximum execution speed.

### Choice 3: Preconditioning & Regularization Handling
* Cotan Laplacian $L$ has a 1-dimensional null space (constant functions). `geometry-central` regularizes Poisson with $10^{-6} I$.
* The condition number $\kappa(L + 10^{-6} I)$ is high. Jacobi preconditioning ($M^{-1} = \text{diag}(A)^{-1}$) is $O(N)$ memory and embarrassingly parallel on GPU.
* Stopping criterion: relative residual $\|r_k\|_2 / \|b\|_2 \le \text{tol}$ (default $\text{tol} = 10^{-6}$), which yields visually identical distance fields while avoiding unnecessary iterations.

### Choice 4: Opt-in CMake Architecture
* Preserve upstream build integrity by controlling CUDA via `option(GC_ENABLE_CUDA "Enable CUDA backend" OFF)`.
* Only search for CUDA toolkit and link cuSPARSE/cuBLAS when `GC_ENABLE_CUDA` is `ON`.

---

## Inspected Source Files
The following files were inspected for this analysis:
1. `include/geometrycentral/numerical/linear_solvers.h`
2. `src/numerical/linear_solvers.cpp`
3. `src/numerical/positive_definite_solvers.cpp`
4. `include/geometrycentral/numerical/linear_algebra_types.h`
5. `include/geometrycentral/surface/heat_method_distance.h`
6. `src/surface/heat_method_distance.cpp`

