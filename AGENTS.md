# AGENTS.md: GPU-Accelerated Extension of geometry-central

Persistent operational context, architectural reference, and guidelines for AI agents and developers working on this codebase.

> **CRITICAL RULE**: Before modifying a file, inspect only the relevant function, class, or surrounding code. Do not reread the entire repository unless explicitly requested.

---

## 1. Project Objective
Add a high-performance CUDA GPU backend for sparse Symmetric Positive-Definite (SPD) linear systems to `geometry-central`:
1. Implement a GPU-accelerated Jacobi-preconditioned Conjugate Gradient (PCG) solver using cuSPARSE and cuBLAS.
2. Integrate cleanly with existing `LinearSolver<T>` abstractions without breaking CPU functionality.
3. Accelerate the Heat Method for geodesic distance (`HeatMethodDistanceSolver`) and Vector Heat Method (`VectorHeatMethodSolver`).
4. Benchmark and compare CPU (SuiteSparse CHOLMOD direct $LDL^T$ and Eigen `SimplicialLDLT`) vs. GPU PCG across small to large meshes ($> 250\text{k}$ vertices).
5. Ensure robustness on non-manifold and degenerate CAD geometries (leveraging intrinsic Delaunay mollification and tufted covers).

---

## 2. Upstream Repository
* **Repository**: `geometry-central` (C++11 geometry-processing library).
* **Core Dependencies**: Eigen 3.3+, SuiteSparse (CHOLMOD, UMFPACK), GoogleTest.
* **Working Directory**: `/DATA/suraj/m1/geometry/geometry-central`

---

## 3. Current Architecture Summary
* **Base Solver Interface**: `LinearSolver<T>` in `include/geometrycentral/numerical/linear_solvers.h` declares pure virtual `solve(rhs) -> Vector<T>` and `solve(x, rhs) -> void`.
* **Current SPD Solver**: `PositiveDefiniteSolver<T>` in `src/numerical/positive_definite_solvers.cpp` uses the PIMPL pattern (`PSDSolverInternals<T>`).
  * Direct simplicial $LDL^T$ factorization via SuiteSparse CHOLMOD (if `GC_HAVE_SUITESPARSE`).
  * Fallback to `Eigen::SimplicialLDLT`.
* **Matrix Layout**: `SparseMatrix<T>` is `Eigen::SparseMatrix<T, Eigen::ColMajor>` (Compressed Sparse Column / CSC).
  * **Key Property**: For symmetric matrices ($A = A^T$), CSC layout is structurally and numerically identical to Compressed Sparse Row (CSR) format: `outerIndexPtr()` $\equiv$ `csrRowOffsets`, `innerIndexPtr()` $\equiv$ `csrColInd`, `valuePtr()` $\equiv$ `csrValues`. This enables zero-copy host-to-device transfers.
* **Solver Lifecycle in Algorithms**: `HeatMethodDistanceSolver` and `VectorHeatMethodSolver` construct solvers as member `std::unique_ptr` instances. Factorization is performed once in the constructor/init, and forward/backward solves are reused across queries.

---

## 4. Relevant File Paths

Detailed descriptions and dependency graphs are documented in [`docs/gpu_solver/FILE_MAP.md`](docs/gpu_solver/FILE_MAP.md).

* **Solver Interfaces & CPU Implementations**:
  * `include/geometrycentral/numerical/linear_solvers.h` (Base `LinearSolver<T>`, `PositiveDefiniteSolver<T>`)
  * `src/numerical/linear_solvers.cpp` (Explicit template instantiations, `residual()`)
  * `src/numerical/positive_definite_solvers.cpp` (CHOLMOD and Eigen LDLT implementations)
  * `src/numerical/square_solvers.cpp` (UMFPACK / Eigen SparseLU for non-SPD systems)
* **Linear Algebra Core**:
  * `include/geometrycentral/numerical/linear_algebra_types.h` (`Vector<T>`, `SparseMatrix<T>`)
  * `include/geometrycentral/numerical/linear_algebra_utilities.h` & `.ipp` (Utilities, sanity checks)
  * `include/geometrycentral/numerical/suitesparse_utilities.h` & `src/numerical/suitesparse_utilities.cpp` (CHOLMOD interop)
* **Downstream Algorithms**:
  * `include/geometrycentral/surface/heat_method_distance.h` & `src/surface/heat_method_distance.cpp` (Heat method geodesic distance)
  * `include/geometrycentral/surface/vector_heat_method.h` & `src/surface/vector_heat_method.cpp` (Vector transport, log maps)
* **Geometry & Robust Operators**:
  * `include/geometrycentral/surface/intrinsic_geometry_interface.h` (`cotanLaplacian`, `vertexLumpedMassMatrix`)
  * `include/geometrycentral/surface/tufted_laplacian.h` (Intrinsic mollification & tufted cover for degenerate/non-manifold meshes)
* **Build System & Tests**:
  * `CMakeLists.txt`, `deps/CMakeLists.txt`, `src/CMakeLists.txt`
  * `test/CMakeLists.txt`, `test/src/linear_algebra_test.cpp`
  * `include/geometrycentral/utilities/timing.h` (Benchmarking macros)
* **GPU Project Documentation**:
  * `docs/gpu_solver/ARCHITECTURE.md`
  * `docs/gpu_solver/FILE_MAP.md`
  * `docs/gpu_solver/INTEGRATION_PLAN.md`

---

## 5. Coding Conventions
* **Language Standard**: C++11 for host code (`cxx_std_11`); CUDA C++14/17 for device code (`.cu`).
* **Formatting**: Defined in `.clang-format`:
  * 2-space indentation, no tabs.
  * 120 character column limit.
  * Attached braces (`BreakBeforeBraces: Attach`).
  * Left pointer alignment (`T* ptr`).
* **Namespaces**: All core symbols reside in `namespace geometrycentral` or sub-namespaces (e.g., `geometrycentral::surface`).
* **PIMPL & Encapsulation**: Isolate external headers (cuSPARSE, cuBLAS, CUDA runtime) in private internal structs or `.cu`/`.cpp` files to avoid polluting public headers.
* **Non-Destructive Changes**: All CPU functionality must remain fully functional. If CUDA is disabled (`GC_ENABLE_CUDA=OFF`), the library must compile and behave identically to upstream.

---

## 6. GPU Backend Design Constraints
* **Algorithm**: Preconditioned Conjugate Gradient (PCG) with Jacobi (diagonal) scaling:
  * Preconditioner: $M^{-1} = \text{diag}(A)^{-1}$ stored as a device vector $d_{\text{inv}}[i] = 1.0 / A_{ii}$.
* **GPU Stack**:
  * cuSPARSE for sparse matrix-vector product (`cusparseSpMV`).
  * cuBLAS for vector updates (`cublasDaxpy`, `cublasDdot`, `cublasDnrm2`) or lightweight fused CUDA kernels.
  * CUDA Runtime for device memory management (`cudaMalloc`, `cudaMemcpy`, streams).
* **Buffer Allocation**: Pre-allocate matrix CSR arrays and working vectors ($x, r, z, p, q, d_{\text{inv}}$) during solver setup; avoid any `cudaMalloc`/`cudaFree` during `solve()` invocations.
* **Precision**: Target `double` precision as default (`CUDA_R_64F`, `double`) to prevent numerical error on ill-conditioned or unnormalized geometry. Provide `float` as an optional template instantiation.
* **CMake Opt-in**: Controlled via `option(GC_ENABLE_CUDA "Enable CUDA GPU acceleration" OFF)`.

---

## 7. Testing Requirements
* Built on GoogleTest via `test/CMakeLists.txt`.
* **Solver Unit Tests**:
  * Synthetic SPD systems (e.g., random SPD graph Laplacians as in `test/src/linear_algebra_test.cpp`).
  * Convergence validation: residual $\|A x - b\|_2 < 10^{-4}$ (or relative residual $< 10^{-6}$).
  * Consistency: assert GPU PCG output matches CPU direct solver output within tolerance.
* **Heat Method Tests**:
  * Verify geodesic distance computations on standard assets (`spot.ply`, `bob_small.ply`).
  * Compare distance field values between CPU and GPU solvers.

---

## 8. Benchmarking Requirements
* **Metrics**:
  * Factorization / setup time vs. per-solve time.
  * Host-to-device and device-to-host memory transfer times.
  * Total time per query across CPU direct (CHOLMOD), CPU fallback (Eigen), and GPU PCG.
* **Scaling Profiles**:
  * Small meshes ($< 10\text{k}$ vertices) to measure PCIe launch latency crossover point.
  * Medium meshes ($50\text{k} - 100\text{k}$ vertices).
  * Large meshes ($> 250\text{k}$ - millions of vertices) where GPU throughput dominates.
* **Tooling**: Use `geometrycentral/utilities/timing.h` and CUDA event timing (`cudaEventRecord`, `cudaEventElapsedTime`).

---

## 9. Known Risks & Mitigations
1. **Poisson Operator Ill-Conditioning**: Pure cotan Laplacian has a 1D null space. Shifting by $10^{-6} I$ yields a large condition number $\kappa(L + 10^{-6} I)$.
   * *Mitigation*: Tune relative residual stopping threshold ($\text{tol} \approx 10^{-6}$); monitor PCG iteration counts.
2. **Degenerate Meshes (Negative Cotan Weights)**: Obsolete/obtuse CAD triangles can cause the Laplacian to lose positive-definiteness.
   * *Mitigation*: Utilize `useRobustLaplacian = true` mode in `HeatMethodDistanceSolver` (applies intrinsic mollification and Delaunay flips to guarantee non-negative weights).
3. **PCIe Overhead on Small Meshes**: Transfer latency may exceed CPU solve time for meshes under $10\text{k}$ vertices.
   * *Mitigation*: Benchmark crossover point and clearly document recommended mesh sizes for GPU acceleration.
4. **Complex Linear Systems**: `VectorHeatMethodSolver` uses `std::complex<double>` for connection Laplacians.
   * *Mitigation*: Phase 1 implements real SPD solvers; Phase 2 implements complex Hermitian PCG (`CUDA_C_64F`).

---

## 10. Current Implementation Status
* **Exploration Phase**: Complete.
* **Documentation**: `docs/gpu_solver/ARCHITECTURE.md`, `docs/gpu_solver/FILE_MAP.md`, `docs/gpu_solver/INTEGRATION_PLAN.md` created.
* **Source Code**: No upstream code modified; working tree clean.
* **Next Step**: Configure CMake for CUDA (`GC_ENABLE_CUDA`) and implement the core `CUDAPCGPositiveDefiniteSolver` class with cuSPARSE and cuBLAS.

