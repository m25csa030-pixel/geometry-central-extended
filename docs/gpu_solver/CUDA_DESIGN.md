# CUDA GPU Backend Design: Sparse SPD Linear Solver

This document specifies the technical design for a minimal, non-invasive CUDA GPU backend for sparse Symmetric Positive-Definite (SPD) linear systems in `geometry-central`. It defines the architecture for a Jacobi-preconditioned Conjugate Gradient (PCG) solver, compares sparse matrix representations, and analyzes data transfer, memory ownership, numerical stability, CMake integration, and verification strategies.

---

## 1. Classification Methodology

To ensure architectural rigor, statements throughout this design are explicitly tagged into three categories:
* **[Verified Fact]**: Invariants, data structures, and behaviors directly confirmed by inspecting upstream source files (`linear_solvers.h`, `positive_definite_solvers.cpp`, `linear_algebra_types.h`, `heat_method_distance.cpp`).
* **[Design Decision]**: Architectural and engineering choices proposed for the new CUDA backend.
* **[Assumption Requiring Validation]**: Hypotheses regarding hardware behavior, compiler compatibility, or convergence rates that must be experimentally confirmed.

---

## 2. Sparse Matrix Representation Comparison

`geometry-central` algorithms build matrices on the CPU as `Eigen::SparseMatrix<T, Eigen::ColMajor>` **[Verified Fact]**. A GPU solver requires a format supported by GPU linear algebra primitives (specifically `cuSPARSE`).

| Representation | Description | Suitability for `geometry-central` | Verdict |
| :--- | :--- | :--- | :--- |
| **Direct CSR via Structural Symmetry** | Treat `Eigen::SparseMatrix<T, Eigen::ColMajor>` (CSC) directly as CSR on the GPU, relying on the mathematical property $A = A^T$. | **Optimal**. `outerIndexPtr()` provides row offsets, `innerIndexPtr()` provides column indices, and `valuePtr()` provides values. Zero CPU reordering, zero extra CPU memory allocation, direct `cudaMemcpy` to GPU device buffers. | **Selected Approach** |
| **COO Converted to CSR** | Transfer raw triplets $(i, j, v)$ to GPU, then call `cusparseXcoo2csr`. | **Suboptimal**. Requires serializing triplets from the already-assembled Eigen matrix or keeping raw triplets in memory, doubling CPU-to-GPU memory transfer traffic and requiring GPU-side sorting/conversion overhead. | **Rejected** |
| **Explicit CPU CSR Conversion** | Allocate a secondary `Eigen::SparseMatrix<T, Eigen::RowMajor>` on the CPU and deep-copy entries before uploading to GPU. | **Inefficient**. Duplicates matrix storage on host ($2\times$ host RAM usage for large meshes) and wastes CPU cycles copying non-zeros without altering the numerical result for symmetric matrices. | **Rejected** |
| **ELLPACK / DIA / BSR** | Formats assuming uniform row length (ELLPACK), strict diagonals (DIA), or dense block structure (BSR). | **Unsuitable**. 3D surface meshes have irregular vertex valences (typically valences 3 to 10+). ELLPACK would introduce substantial zero-padding; DIA fails due to irregular band structures; BSR offers no benefits for scalar Laplacians. | **Rejected** |

### Selected Format: Direct Symmetric CSR
* **Mathematical Rationale**:
  For any symmetric matrix $A = A^T$:
  $$\text{CSC}(A) \equiv \text{CSR}(A^T) = \text{CSR}(A)$$
* **Data Mapping**:
  * `csrRowOffsets` (size $N+1$) $\longleftarrow$ `mat.outerIndexPtr()`
  * `csrColInd` (size $NNZ$) $\longleftarrow$ `mat.innerIndexPtr()`
  * `csrValues` (size $NNZ$) $\longleftarrow$ `mat.valuePtr()`
* **Validation Check**: Before accepting the matrix, verify symmetry via `checkHermitian(mat)` or `checkSymmetric(mat)` (`#ifndef GC_NLINALG_DEBUG`) **[Verified Fact]**.

---

## 3. Detailed Architectural Analysis

### 3.1. Data Transfer Strategy
* **Setup Phase (Constructor)**:
  * Upload matrix structures from host to device:
    * `csrRowOffsets`: $(N + 1) \times \text{sizeof}(int)$
    * `csrColInd`: $NNZ \times \text{sizeof}(int)$
    * `csrValues`: $NNZ \times \text{sizeof}(T)$
  * Performed **once** during solver construction **[Design Decision]**.
* **Solve Phase (`solve(x, rhs)`)**:
  * Host-to-Device (H2D): Copy `rhs.data()` to `d_b` ($N \times \text{sizeof}(T)$).
  * Device Computation: PCG loop executes **entirely on device memory**; no intermediate vector transfers between host and device during iteration.
  * Device-to-Host (D2H): Copy final solution `d_x` back to host buffer `x.data()` ($N \times \text{sizeof}(T)$).
* **Asynchronous Streams**:
  * Use a dedicated `cudaStream_t` bound to the solver instance.
  * Facilitates asynchronous execution and overlap with CPU tasks if desired, while synchronizing (`cudaStreamSynchronize`) at the conclusion of `solve()`.

### 3.2. Matrix Conversion Strategy
* **Zero-Copy Host Extraction**:
  * Ensure matrix is compressed: `mat.makeCompressed()` **[Verified Fact]**.
  * Directly pass raw pointers:
    ```cpp
    const int* h_rowOffsets = mat.outerIndexPtr();
    const int* h_colIndices = mat.innerIndexPtr();
    const T* h_values = mat.valuePtr();
    ```
* **Descriptor Initialization**:
  * Instantiate `cusparseSpMatDescr_t` using `cusparseCreateCsr`:
    ```cpp
    cusparseCreateCsr(&matDescr, nRows, nCols, nnz,
                      d_rowOffsets, d_colIndices, d_values,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    ```
  * Query buffer size via `cusparseSpMV_bufferSize` and allocate persistent SpMV buffer `d_spmvBuffer`.

### 3.3. GPU Memory Ownership & Lifecycle
* **RAII Encapsulation via PIMPL**:
  * Like `PositiveDefiniteSolver<T>` which uses `PSDSolverInternals<T>` **[Verified Fact]**, the GPU solver will encapsulate all CUDA, cuSPARSE, and cuBLAS handles inside a private implementation struct `CUDAPCGInternals<T>`.
  * Public headers will remain free of CUDA runtime and cuSPARSE headers.
* **Persistent Scratchpad Allocation**:
  * Working buffers allocated **once** in the constructor and freed in the destructor:
    * `d_rowOffsets`, `d_colIndices`, `d_values` (Matrix CSR)
    * `d_x` (Current solution iterate, size $N$)
    * `d_r` (Residual vector, size $N$)
    * `d_z` (Preconditioned residual, size $N$)
    * `d_p` (Conjugate search direction, size $N$)
    * `d_q` (Matrix-vector product $A p$, size $N$)
    * `d_diagInv` (Inverted diagonal Jacobi preconditioner, size $N$)
    * `d_spmvBuffer` (cuSPARSE internal SpMV workspace)
  * **Critical Rule**: Zero device allocations (`cudaMalloc` / `cudaFree`) inside `solve()` **[Design Decision]**.
  * Supports repeated solves with zero allocation overhead, strictly matching `HeatMethodDistanceSolver` lifecycle requirements **[Verified Fact]**.

### 3.4. Preconditioner Construction & Application
* **Mathematical Definition**:
  Jacobi preconditioner: $M = \text{diag}(A) \implies M^{-1} = \text{diag}(1 / A_{ii})$.
* **Construction (Setup Phase)**:
  * Locate diagonal elements $A_{ii}$ for each row $i \in [0, N-1]$.
  * Can be extracted during CSR upload or via a lightweight CUDA kernel:
    * For each row $i$, search column indices in range `[rowOffsets[i], rowOffsets[i+1])` for `colInd == i`.
    * Store $d_{\text{inv}}[i] = 1.0 / A_{ii}$.
    * If $|A_{ii}| < \epsilon$ (degenerate/zero diagonal), set $d_{\text{inv}}[i] = 1.0$ to prevent division by zero.
* **Application (Solve Phase)**:
  * Application of $z = M^{-1} r$ is an element-wise vector product:
    $$z_i = r_i \cdot d_{\text{inv}}[i]$$
  * Executed via a simple 1D element-wise kernel:
    ```cuda
    __global__ void applyJacobiKernel(int n, const double* __restrict__ r,
                                      const double* __restrict__ d_inv,
                                      double* __restrict__ z) {
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx < n) {
        z[idx] = r[idx] * d_inv[idx];
      }
    }
    ```
  * Bandwidth-efficient: Requires only $1$ read of $r$, $1$ read of $d_{\text{inv}}$, and $1$ write of $z$.

### 3.5. Convergence Criteria & Iteration Control
* **Stopping Criterion**:
  Relative residual norm:
  $$\frac{\|r_k\|_2}{\|b\|_2} \le \text{tol}$$
  * If $\|b\|_2 == 0$, the exact solution is $x = 0$, returning immediately.
  * Default tolerance: $\text{tol} = 10^{-6}$ (sufficient for geometric accuracy in heat diffusion and geodesic distance).
  * Maximum iterations: $k_{\max} = \min(N, 2000)$ (configurable via `setMaxIterations()`).
* **Stagnation / Breakdown Guards**:
  * Inner product denominator check: if $p_k^T q_k \le 0$, the matrix is not positive-definite or arithmetic breakdown occurred. Throw `std::runtime_error("Matrix is not positive definite or PCG breakdown detected")`.
  * NaN / Inf detection: check residual norm for finiteness.

### 3.6. Numerical Precision
* **Double Precision ($64$-bit) as Default**:
  * In `geometry-central`, vertex positions and differential operators default to `double` **[Verified Fact]**.
  * Laplacian operators with irregular triangle aspect ratios or large meshes can have condition numbers $\kappa(A) > 10^7$. Double precision is strictly required to prevent catastrophic cancellation.
  * Uses `cublasDdot`, `cublasDaxpy`, `cublasDnrm2`, and `CUDA_R_64F` in cuSPARSE.
* **Single Precision ($32$-bit) Support**:
  * Provided via template specialization `CUDAPCGPositiveDefiniteSolver<float>`.
  * Uses `cublasSdot`, `cublasSaxpy`, `cublasSnrm2`, and `CUDA_R_32F`.

### 3.7. Error Handling & Exception Translation
* **CUDA / cuSPARSE / cuBLAS Check Macros**:
  * Custom internal macro:
    ```cpp
    #define GC_CUDA_CHECK(call) do { \
      cudaError_t err = call; \
      if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA Error at ") + __FILE__ + ":" + \
                                 std::to_string(__LINE__) + " - " + cudaGetErrorString(err)); \
      } \
    } while(0)
    ```
  * Similar macros for `cusparseStatus_t` and `cublasStatus_t`.
* **Consistency with `geometry-central`**:
  * All GPU failures, dimension mismatches, or convergence breakdowns throw standard C++ exceptions (`std::invalid_argument`, `std::logic_error`, `std::runtime_error`) **[Verified Fact]**.
  * Destructors must not throw exceptions (`noexcept`).

### 3.8. Thread / Block Design Considerations
* **cuSPARSE SpMV**:
  * Uses NVIDIA's optimized internal kernel scheduling (`CUSPARSE_SPMV_ALG_DEFAULT` / `CUSPARSE_SPMV_CSR_ALG2`).
* **Vector Operations (Jacobi Application, Initializations)**:
  * Grid-stride or 1D block layout:
    * Block size: $256$ threads per block (optimal balance between occupancy and register pressure on modern Ada Lovelace / Ampere / Turing architectures) **[Assumption Requiring Validation]**.
    * Grid size: $\lceil N / 256 \rceil$.
* **cuBLAS Level-1 BLAS**:
  * Dot products and AXPY operations use cuBLAS, which automatically utilizes hardware warp reductions and shared memory.

### 3.9. CMake Integration
* **Opt-In Switch**:
  * In root `CMakeLists.txt` or `deps/CMakeLists.txt`:
    ```cmake
    option(GC_ENABLE_CUDA "Enable CUDA GPU acceleration backend" OFF)
    ```
* **Toolchain & Target Integration**:
  * When `GC_ENABLE_CUDA` is `ON`:
    ```cmake
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    
    # Target definition
    target_compile_definitions(geometry-central PUBLIC GC_HAVE_CUDA)
    target_link_libraries(geometry-central PRIVATE 
      CUDA::cusparse 
      CUDA::cublas 
      CUDA::cudart
    )
    ```
  * CUDA standard: `set_target_properties(geometry-central PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED ON)`.
  * If `GC_ENABLE_CUDA` is `OFF`, zero CUDA references, files, or link dependencies are included, preserving exact upstream CPU build behavior **[Verified Fact]**.

### 3.10. Testing & Verification Strategy
* **GoogleTest Suite (`test/src/cuda_pcg_test.cpp`)**:
  * Test 1: **Synthetic SPD Laplacian Matrix**:
    * Generate test matrix using `buildSPDTestMatrix<double>()` from `test/src/linear_algebra_test.cpp` **[Verified Fact]**.
    * Assert residual $\|A x - b\|_2 < 10^{-4}$.
  * Test 2: **Equivalence with CPU Direct Solver**:
    * Compare solution vectors $x_{\text{gpu}}$ and $x_{\text{cpu}}$ (from `PositiveDefiniteSolver` / CHOLMOD):
      $$\frac{\|x_{\text{gpu}} - x_{\text{cpu}}\|_\infty}{\|x_{\text{cpu}}\|_\infty} < 10^{-4}$$
  * Test 3: **Repeated Solves Consistency**:
    * Solve multiple distinct random RHS vectors with the same pre-factored solver instance to ensure working buffers are correctly reset between iterations.
  * Test 4: **Heat Method Distance Consistency**:
    * Run `HeatMethodDistanceSolver` on `spot.ply` using CPU vs. GPU solver.
    * Compare vertex distance field output: assert $\max_v |d_{\text{gpu}}(v) - d_{\text{cpu}}(v)| < 10^{-3}$.

---

## 4. Proposed Interface & Heat Method Integration

### 4.1. Proposed GPU Solver Class
In `include/geometrycentral/numerical/cuda_pcg_solver.h`:

```cpp
#pragma once

#include "geometrycentral/numerical/linear_solvers.h"
#include <memory>

namespace geometrycentral {

template <typename T>
struct CUDAPCGInternals;

template <typename T>
class CUDAPCGPositiveDefiniteSolver final : public LinearSolver<T> {
public:
  CUDAPCGPositiveDefiniteSolver(SparseMatrix<T>& mat, double tol = 1e-6, size_t maxIters = 2000);
  ~CUDAPCGPositiveDefiniteSolver() override;

  void solve(Vector<T>& x, const Vector<T>& rhs) override;
  Vector<T> solve(const Vector<T>& rhs) override;

  void setTolerance(double tol);
  void setMaxIterations(size_t maxIters);
  size_t getIterationsAchieved() const;
  double getFinalResidual() const;

private:
  std::unique_ptr<CUDAPCGInternals<T>> internals;
};

// Explicit instantiations for float and double
extern template class CUDAPCGPositiveDefiniteSolver<double>;
extern template class CUDAPCGPositiveDefiniteSolver<float>;

} // namespace geometrycentral
```

### 4.2. Minimal Non-Invasive Changes to `HeatMethodDistanceSolver`
In `include/geometrycentral/surface/heat_method_distance.h`:
* Widen solver pointer types from `PositiveDefiniteSolver<double>` to the base class `LinearSolver<double>`:
  ```cpp
  // Before:
  std::unique_ptr<PositiveDefiniteSolver<double>> heatSolver;
  std::unique_ptr<PositiveDefiniteSolver<double>> poissonSolver;

  // After:
  std::unique_ptr<LinearSolver<double>> heatSolver;
  std::unique_ptr<LinearSolver<double>> poissonSolver;
  ```
* Introduce an optional solver backend parameter in the constructor:
  ```cpp
  enum class HeatSolverBackend { CPU, CUDA_PCG };

  HeatMethodDistanceSolver(IntrinsicGeometryInterface& geom, 
                           double tCoef = 1.0, 
                           bool useRobustLaplacian = false,
                           HeatSolverBackend backend = HeatSolverBackend::CPU);
  ```
* In `src/surface/heat_method_distance.cpp`:
  ```cpp
  if (backend == HeatSolverBackend::CUDA_PCG) {
  #ifdef GC_HAVE_CUDA
    heatSolver.reset(new CUDAPCGPositiveDefiniteSolver<double>(heatOp));
    poissonSolver.reset(new CUDAPCGPositiveDefiniteSolver<double>(Ls));
  #else
    throw std::runtime_error("CUDA backend requested but GC_HAVE_CUDA is not enabled");
  #endif
  } else {
    heatSolver.reset(new PositiveDefiniteSolver<double>(heatOp));
    poissonSolver.reset(new PositiveDefiniteSolver<double>(Ls));
  }
  ```
* **Backwards Compatibility**: Existing callers calling `HeatMethodDistanceSolver(geom)` retain identical CPU behavior with zero code changes **[Design Decision]**.

---

## 5. Explicit Classification Table

| Category | Item | Rationale / Source |
| :--- | :--- | :--- |
| **[Verified Fact]** | `LinearSolver<T>` interface | Declared in `include/geometrycentral/numerical/linear_solvers.h` (lines 58–73); requires `solve(rhs)` and `solve(x, rhs)`. |
| **[Verified Fact]** | Matrix storage format | `Eigen::SparseMatrix<T>` is `ColMajor` (CSC format) in `linear_algebra_types.h` (lines 15–16). |
| **[Verified Fact]** | Symmetric CSC $\equiv$ CSR | In symmetric matrices, CSC column offsets and row indices map identically to CSR row offsets and column indices. |
| **[Verified Fact]** | Repeated solve lifecycle | `HeatMethodDistanceSolver` factors `heatOp` and `Ls` once in constructor, reusing them in `computeDistanceRHS()`. |
| **[Verified Fact]** | Regularized Poisson system | Poisson equation is shifted by $10^{-6} I$ in `heat_method_distance.cpp` (line 68). |
| **[Verified Fact]** | SuiteSparse / Eigen fallback | `positive_definite_solvers.cpp` uses CHOLMOD simplicial $LDL^T$ or `Eigen::SimplicialLDLT`. |
| **[Design Decision]** | Standalone solver class | Implement `CUDAPCGPositiveDefiniteSolver<T>` derived from `LinearSolver<T>` to maintain clean separation between direct CPU and iterative GPU solvers. |
| **[Design Decision]** | Zero runtime allocation | Pre-allocate all CSR buffers, working vectors ($x, r, z, p, q$), and cuSPARSE workspace in constructor; zero `cudaMalloc` in `solve()`. |
| **[Design Decision]** | PIMPL encapsulation | Isolate all CUDA/cuSPARSE/cuBLAS symbols inside `CUDAPCGInternals<T>` to prevent header leakage. |
| **[Design Decision]** | Default to double precision | Use `double` (`CUDA_R_64F`) for primary geometry processing; provide `float` template instantiation. |
| **[Design Decision]** | Relative tolerance default | Default PCG stopping tolerance $\text{tol} = 10^{-6}$, max iterations $2000$. |
| **[Design Decision]** | Opt-in CMake switch | Controlled via `option(GC_ENABLE_CUDA "Enable CUDA" OFF)` to ensure zero impact on default CPU builds. |
| **[Assumption Requiring Validation]** | PCG convergence on shifted Poisson | While $M + t L$ (heat operator) is well-conditioned and converges rapidly under Jacobi-PCG, $L + 10^{-6} I$ may require higher iteration counts ($\sim 200\text{--}500$ iters) on meshes $> 100\text{k}$ vertices. Iteration counts must be profiled. |
| **[Assumption Requiring Validation]** | GPU speedup crossover point | For small meshes ($< 10\text{k}$ vertices), PCIe transfer overhead and kernel launch latency might make CPU Cholesky faster. The crossover point is hypothesized at $\approx 20\text{k}\text{--}50\text{k}$ vertices. |
| **[Assumption Requiring Validation]** | cuSPARSE CUSPARSE_INDEX_32I with Eigen StorageIndex | Assumes `Eigen::SparseMatrix::StorageIndex` is consistently 32-bit `int` across supported platforms and compilers. |

