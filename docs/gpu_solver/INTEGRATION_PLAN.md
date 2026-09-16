# Integration Plan: CUDA GPU Backend for Sparse SPD Solvers

This document describes the design, implementation roadmap, minimal required changes, potential risks, and open technical questions for adding a CUDA GPU backend with a Jacobi-preconditioned Conjugate Gradient (PCG) solver to `geometry-central`.

---

## 1. Proposed CUDA Backend Design

### 1.1. Preconditioned Conjugate Gradient (PCG) Algorithm
For a sparse symmetric positive-definite linear system $A x = b$ with symmetric positive-definite preconditioner $M$:

$$\begin{aligned}
r_0 &= b - A x_0 \\
z_0 &= M^{-1} r_0 \\
p_0 &= z_0 \\
\gamma_0 &= r_0^T z_0
\end{aligned}$$

For $k = 0, 1, 2, \dots$ until convergence ($\|r_k\|_2 / \|b\|_2 \le \text{tol}$ or $k \ge k_{\max}$):
$$\begin{aligned}
q_k &= A p_k && \text{[SpMV: Sparse Matrix-Vector Multiply]} \\
\alpha_k &= \frac{\gamma_k}{p_k^T q_k} && \text{[Dot Product]} \\
x_{k+1} &= x_k + \alpha_k p_k && \text{[AXPY]} \\
r_{k+1} &= r_k - \alpha_k q_k && \text{[AXPY]} \\
z_{k+1} &= M^{-1} r_{k+1} && \text{[Preconditioner Solve]} \\
\gamma_{k+1} &= r_{k+1}^T z_{k+1} && \text{[Dot Product]} \\
\beta_k &= \frac{\gamma_{k+1}}{\gamma_k} \\
p_{k+1} &= z_{k+1} + \beta_k p_k && \text{[AXPBY]}
\end{aligned}$$

### 1.2. Preconditioner: Jacobi (Diagonal Scaling)
* **Definition**: $M = \text{diag}(A)$ where $M_{ii} = A_{ii}$.
* **Application**: $z_i = M^{-1} r_i = r_i / A_{ii}$.
* **GPU Characteristics**:
  * Memory: $O(N)$ storage for the inverted diagonal array $d_{\text{inv}}[i] = 1.0 / A_{ii}$.
  * Execution: Single embarrassingly parallel element-wise kernel: $z[i] = r[i] \cdot d_{\text{inv}}[i]$.
  * Setup: Extracted once in $O(N)$ during solver initialization by inspecting the diagonal elements in the CSR format.

### 1.3. GPU Library Stack
* **CUDA Runtime (`cudaMalloc`, `cudaMemcpy`, `cudaStream_t`)**: Buffer allocation and asynchronous execution.
* **cuSPARSE**:
  * Matrix representation: `cusparseSpMatDescr_t` wrapping CSR pointers (`csrRowOffsets`, `csrColInd`, `csrValues`).
  * Matrix-vector product: `cusparseSpMV()` using modern generic API (`CUSPARSE_SPMV_ALG_DEFAULT`).
* **cuBLAS**:
  * Vector operations: `cublasDdot` (inner products), `cublasDaxpy` (vector updates), `cublasDnrm2` (residual norm).
  * Alternatively, lightweight custom fused CUDA kernels can compute $x$ and $r$ updates simultaneously to minimize memory bandwidth.

### 1.4. Lifetime & Memory Architecture
To avoid expensive GPU allocation during `solve()` calls:
1. **Construction / Factorization**:
   * Allocates device arrays for CSR matrix (`d_rowOffsets`, `d_colIndices`, `d_values`).
   * Allocates working vectors (`d_x`, `d_r`, `d_z`, `d_p`, `d_q`).
   * Extracts and inverts diagonal elements to populate `d_diagInv`.
   * Allocates cuSPARSE `cusparseSpMV` workspace buffer.
2. **Execution (`solve`)**:
   * Copies RHS vector $b$ to device (`cudaMemcpyHostToDevice`).
   * Executes iterative PCG loop entirely within GPU device memory.
   * Copies result $x$ back to host (`cudaMemcpyDeviceToHost`).
3. **Destruction**:
   * Cleans up device arrays, cuSPARSE descriptors, and cuBLAS handles.

---

## 2. Minimal Required Changes

To preserve existing CPU functionality and adhere strictly to repository safety, changes will be minimal, non-invasive, and opt-in.

### 2.1. CMake Configuration
* In top-level `CMakeLists.txt` or `deps/CMakeLists.txt`:
  ```cmake
  option(GC_ENABLE_CUDA "Enable CUDA GPU acceleration" OFF)
  if(GC_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    # Target compile definitions and include directories
  endif()
  ```
* In `src/CMakeLists.txt`:
  * Conditionally include CUDA source files (e.g., `numerical/cuda_pcg_solver.cu`).
  * Link `CUDA::cusparse`, `CUDA::cublas`, `CUDA::cudart` to target `geometry-central`.
  * Add preprocessor define `-DGC_HAVE_CUDA`.

### 2.2. Class Hierarchy & Solver Interface
Two non-breaking options exist:

#### Option A (Recommended): Concrete `CUDAPCGPositiveDefiniteSolver<T>` Implementing `LinearSolver<T>`
* Define `CUDAPCGPositiveDefiniteSolver<T> : public LinearSolver<T>` in `include/geometrycentral/numerical/cuda_pcg_solver.h`.
* Adheres to the existing `LinearSolver<T>` contract:
  ```cpp
  Vector<T> solve(const Vector<T>& rhs) override;
  void solve(Vector<T>& x, const Vector<T>& rhs) override;
  ```
* Leaves `PositiveDefiniteSolver<T>` completely unchanged on CPU.

#### Option B: Unified `PositiveDefiniteSolver<T>` with Backend Selector
* Introduce an enum `enum class SolverBackend { Default, DirectCPU, CUDA_PCG };`.
* In `PositiveDefiniteSolver<T>`, route to `PSDSolverInternals` (CHOLMOD/Eigen) or a CUDA backend pointer based on the selector.

### 2.3. Integration into Heat Method
In `HeatMethodDistanceSolver` (`include/geometrycentral/surface/heat_method_distance.h`):
* Introduce an optional configuration parameter in the constructor:
  ```cpp
  enum class HeatSolverBackend { CPU, CUDA_PCG };
  HeatMethodDistanceSolver(IntrinsicGeometryInterface& geom,
                           double tCoef = 1.0,
                           bool useRobustLaplacian = false,
                           HeatSolverBackend backend = HeatSolverBackend::CPU);
  ```
* Change `heatSolver` and `poissonSolver` member types from:
  ```cpp
  std::unique_ptr<PositiveDefiniteSolver<double>> heatSolver;
  ```
  to the base class:
  ```cpp
  std::unique_ptr<LinearSolver<double>> heatSolver;
  std::unique_ptr<LinearSolver<double>> poissonSolver;
  ```
* Zero impact on existing callers: default argument `HeatSolverBackend::CPU` keeps all existing behavior identical.

### 2.4. Tests and Benchmarking Infrastructure
* Add a dedicated test file: `test/src/cuda_pcg_test.cpp`.
* Add a dedicated benchmark tool: `test/src/benchmark_solvers.cpp` measuring:
  * Setup time vs. Solve time.
  * CPU (SuiteSparse CHOLMOD direct) vs. CPU (Eigen LDLt) vs. GPU (CUDA PCG).
  * Scaling on meshes from 10k to 1M+ vertices.

---

## 3. Risks & Mitigations

| Risk | Impact | Mitigation |
| :--- | :--- | :--- |
| **Poisson Operator Ill-Conditioning** | The pure cotan Laplacian has a 1-dimensional null space (constant functions). While geometry-central shifts the diagonal by $10^{-6} I$, the resulting condition number $\kappa(L + 10^{-6} I)$ remains high on large meshes, potentially slowing PCG convergence. | Test Jacobi preconditioning tolerance thresholds; monitor residual decay; evaluate slightly higher regularization shifts or relative tolerance termination ($\text{tol} = 10^{-6}$ vs $10^{-8}$). |
| **Degenerate / Non-Manifold Triangles** | Bad triangles with obtuse angles produce negative cotangent weights, destroying the positive-definiteness ($M$-matrix property) of the Laplacian and causing PCG breakdown. | Leverage geometry-central's existing `useRobustLaplacian = true` mode, which applies intrinsic mollification (`mollifyIntrinsic`) and Delaunay flips (`flipToDelaunay`), guaranteeing non-negative edge weights. |
| **PCIe Transfer Latency on Small Meshes** | For meshes under $\approx 10{,}000$ vertices, CPU Cholesky backsubstitution takes $< 1\text{ ms}$, while GPU kernel launches and host-to-device transfers could introduce net overhead. | Benchmark across multiple mesh resolutions to establish crossover points; document recommended mesh sizes for GPU acceleration. |
| **Double vs. Single Precision** | Single precision (`float`) executes significantly faster on consumer GPUs, but may suffer cancellation errors in geometric operators on non-uniformly scaled meshes. | Target `double` precision (`cuBLAS` `D` routines and `cuSPARSE` `CUDA_R_64F`) as primary default, with optional `float` template instantiation. |
| **Complex Numbers in Vector Heat Method** | `VectorHeatMethodSolver` requires complex arithmetic for connection Laplacians ($M_{\mathbb{C}} + t L_{\text{conn}}$). | Phase 1 focuses on real SPD solvers (`HeatMethodDistanceSolver` and scalar diffusion). Phase 2 introduces complex Hermitian PCG support (`CUDA_C_64F`). |

---

## 4. Open Questions

1. **Solver Interface Abstraction**: Should the GPU solver be exposed as a standalone class `CUDAPCGPositiveDefiniteSolver<T>` implementing `LinearSolver<T>`, or should `PositiveDefiniteSolver<T>` encapsulate both direct CPU and iterative GPU paths?
   * *Recommendation*: Expose `CUDAPCGPositiveDefiniteSolver<T>` as a distinct `LinearSolver<T>` subclass to maintain clean separation between direct and iterative methods, with helper factories or backend options in downstream algorithms.
2. **Tolerance and Max Iterations Policy**: What stopping criteria should be adopted for PCG in geometry applications?
   * Relative residual $\|r_k\|_2 / \|b\|_2 < 10^{-6}$ is typically sufficient for visually identical geodesic distance fields while cutting iteration count in half compared to $10^{-8}$. Should `tolerance` and `maxIterations` be configurable per solver instance?
3. **End-to-End Pipeline Acceleration**: In the Heat Method, step 2 (gradient computation and divergence evaluation on mesh faces) is currently done on CPU. Should the initial milestone focus on accelerating only the linear solves (`heatSolver` and `poissonSolver`), or should a follow-up milestone implement face gradient/divergence CUDA kernels to avoid copying intermediate fields between host and device?
4. **Benchmark Mesh Corpus**: What large meshes will be used for performance comparison?
   * `test/assets/` currently contains only small meshes (e.g., `spot.ply`, $\sim 3\text{k}$ vertices). A benchmark script capable of loading or generating subdivided spheres/tori or high-resolution Stanford models ($\ge 250\text{k}$ vertices) will be needed for benchmarking.

