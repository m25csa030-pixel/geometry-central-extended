# Final Review: CUDA GPU Backend for geometry-central

**Review Date**: 2026-09-16  

**Scope**: All files created or modified during the GPU backend project.

---

## 1. What Is Implemented

### 1.1. Standalone CUDA PCG Solver (`gpu_solver/`)

A fully self-contained Jacobi-preconditioned Conjugate Gradient solver that does not depend on geometry-central.

| File | Purpose |
|---|---|
| `gpu_solver/include/cuda_pcg_solver.h` | Public API: `CUDASolver<T>`, configuration, diagnostics |
| `gpu_solver/src/cuda_pcg_solver.cu` | cuSPARSE SpMV, cuBLAS BLAS-1 calls, PCG loop |
| `gpu_solver/tests/test_cuda_pcg.cpp` | 13 GoogleTest cases |
| `gpu_solver/CMakeLists.txt` | Standalone build with hardcoded `sm_89` default |

### 1.2. Library Integration (`geometry-central`)

| File | Change |
|---|---|
| `include/geometrycentral/numerical/cuda_pcg_solver.h` | Declares `CUDAPCGPositiveDefiniteSolver<T>` and `ComplexCUDAPCGPositiveDefiniteSolver` |
| `src/numerical/cuda_pcg_solver.cu` | Full implementation (~660 lines); PIMPL via `CUDAPCGInternals<T>` |
| `include/geometrycentral/surface/heat_method_distance.h` | Added `HeatSolverBackend` enum; widened solver members to `LinearSolver<double>` |
| `src/surface/heat_method_distance.cpp` | GPU routing in `ensureHave*Solver()`; `#ifdef GC_HAVE_CUDA` guards |
| `include/geometrycentral/surface/vector_heat_method.h` | Added `VectorHeatSolverBackend` enum; widened solver members |
| `src/surface/vector_heat_method.cpp` | GPU routing for scalar/vector/Poisson solvers; NaN normalization guard |
| `CMakeLists.txt` | `option(GC_ENABLE_CUDA ...)` + `enable_language(CUDA)` |
| `src/CMakeLists.txt` | Conditional `target_sources` for `cuda_pcg_solver.cu` |

### 1.3. Tests and Benchmarks

| File | Content |
|---|---|
| `test/src/heat_method_test.cpp` | 8 GoogleTest cases (1 CPU basic + 4 GPU heat distance + 3 GPU vector heat) |
| `test/CMakeLists.txt` | Adds `heat_method_test.cpp` to `TEST_SRCS` |
| `benchmarks/src/test_heat_method_gpu.cpp` | 4 benchmark cases for heat method GPU |
| `benchmarks/src/test_vector_heat_gpu.cpp` | 5 benchmark cases for vector heat GPU |
| `benchmarks/CMakeLists.txt` | Adds benchmark targets |

### 1.4. Robustness Testing

| File | Content |
|---|---|
| `tests/robustness/test_cad_robustness.cpp` | 5-stage probing driver; covers 7 failure categories |
| `tests/robustness/meshes/*.obj` | 7 synthetic OBJ meshes (non-manifold, degenerate, zero-area, etc.) |
| `docs/gpu_solver/ROBUSTNESS.md` | Analysis of each failure mode |

### 1.5. Documentation

| File | Content |
|---|---|
| `docs/gpu_solver/AGENTS.md` | Operational context, coding conventions, risk register |
| `docs/gpu_solver/ARCHITECTURE.md` | CPU solver flow, matrix representation, integration points |
| `docs/gpu_solver/CUDA_DESIGN.md` | Detailed design spec; verified facts vs. design decisions vs. assumptions |
| `docs/gpu_solver/BENCHMARKING.md` | Empirical benchmark results table and analysis |
| `docs/gpu_solver/ROBUSTNESS.md` | Failure mode catalogue with reproduction steps |

---

## 2. What Is Verified

### 2.1. Correctness

| Claim | Evidence |
|---|---|
| GPU and CPU solvers agree within 1% on `bob_small.ply` heat distances | `CPUvsGPUAgreement` test (PASSED) |
| GPU vector heat scalar extension agrees with CPU within 0.1 | `VectorHeat_ScalarExtension` (PASSED) |
| GPU tangent transport preserves vector magnitude to 1e-4 | `VectorHeat_TangentTransport` (PASSED) |
| GPU log map source value is 0.0 to 1e-4 | `VectorHeat_LogMap` (PASSED) |
| Repeated solves do not corrupt working buffers | `RepeatedSolvesMultipleSources` (5 consecutive solves, PASSED) |
| Full upstream geometry-central test suite unaffected | 209/209 tests pass (reported in session) |

### 2.2. Performance — From Measured Benchmark Data

All numbers below come from `docs/gpu_solver/BENCHMARKING.md` (10-run mean, 3 warmup solves, CUDA event timing on NVIDIA L4, AMD EPYC 7742, CUDA 12.6).

#### Heat Diffusion operator ($M + tL$), FP64, solve-only time:

| Mesh | Vertices | CPU Direct (ms) | GPU PCG (ms) | Speedup |
|---|---|---|---|---|
| spot (base) | 2,930 | 5.32 | 6.05 | 0.88× (GPU slower) |
| spot (subdiv 1x) | 11,714 | 25.72 | 8.21 | **3.1×** |
| spot (subdiv 2x) | 46,850 | 127.54 | 12.60 | **10.1×** |
| spot (subdiv 3x) | 187,394 | 628.90 | 20.58 | **30.5×** |

#### Shifted Poisson operator ($L + 10^{-6}I$), FP64, solve-only time:

| Mesh | Vertices | CPU Direct (ms) | GPU PCG (ms) | Speedup |
|---|---|---|---|---|
| spot (subdiv 1x) | 11,714 | 25.47 | 73.51 | 0.35× (GPU slower) |
| spot (subdiv 2x) | 46,850 | 128.06 | 192.63 | 0.66× (GPU slower) |
| spot (subdiv 3x) | 187,394 | 630.26 | 471.22 | **1.3×** |

> [!NOTE]
> GPU PCG is slower than CPU Cholesky on the Poisson operator for meshes below ~150k vertices because the ill-conditioned shifted Laplacian requires >1,000 PCG iterations. GPU PCG is competitive only at very large scales where CPU Cholesky factorization itself becomes the bottleneck (23+ seconds at 187k vertices).

#### PCIe transfer overhead (heat diffusion, FP64):

- `spot (subdiv 3x)` 187k vertices: transfer = 0.35 ms / total = 20.93 ms = **1.7%**
- Transfer overhead is negligible ($<2\%$) across all tested mesh sizes.

#### FP32 precision:
- FP32 **fails to converge** on the shifted Poisson operator at all mesh sizes (hits 2,000-iteration limit with residual $\approx 10^{-2}$). FP32 is not usable for that system.
- FP32 converges on heat diffusion and runs ~2.5× faster than FP64 at 187k vertices, but introduces larger numerical error vs CPU.

### 2.3. Robustness

| Failure Mode | Behaviour Confirmed |
|---|---|
| Non-manifold edge | `ManifoldSurfaceMesh` throws; `readSurfaceMesh` + robust mode heals it |
| Non-manifold pinch vertex | Throws at Stage 1; same mitigation |
| Self-edge in face list | Throws at Stage 1; robust mode heals via mollification |
| Zero-area collinear face | NaN in matrix (Stage 3); robust mode heals |
| Duplicate coincident vertices | Passes silently; topologically fine |
| Disconnected components | Passes all stages on both CPU and GPU |
| Near-singular Laplacian | GPU PCG converges with zero-mean RHS even at $\epsilon = 0$ |

---

## 3. Review: Issues Found

### 3.1. Unnecessary Code Duplication

**Issue (Low severity):** `BENCHMARKING.md` contains the benchmark data table twice — once in a compact Markdown table (lines 15–96) and once in a LaTeX-formatted version (lines 97–141). Both tables carry identical numerical data with different formatting.

**Recommendation:** Delete the compact duplicate (lines 15–96 of `BENCHMARKING.md`) and keep only the well-formatted LaTeX version. No code change required; documentation only.

---

**Issue (Low severity):** The NVCC path fallback block (`/usr/local/cuda-12.6/bin/nvcc` → `/usr/local/cuda/bin/nvcc`) appears in both `CMakeLists.txt` (root, lines 20–26) and `gpu_solver/CMakeLists.txt` (lines 4–10). This is not a runtime bug but is a maintenance concern.

**Recommendation:** Factor into a shared CMake module or leave as-is given the two files have different build roots.

---

### 3.2. Inconsistent Naming

**Issue (Low severity):** The standalone solver in `gpu_solver/` exposes its public API class under a different name than the integrated solver. The standalone file uses `CUDASolver<T>` (in `gpu_solver/include/cuda_pcg_solver.h`), while the integrated library uses `CUDAPCGPositiveDefiniteSolver<T>`. The two implementations are also independent codebases, not shared.

**Recommendation:** This divergence is intentional (standalone vs. integrated), but a comment in each header clarifying the relationship would prevent future confusion.

---

**Issue (Low severity):** `AGENTS.md` Section 10 still says "Next Step: Complex Hermitian PCG for Vector Heat Method (`VectorHeatMethodSolver`)." This step was completed. The status text is stale.

**Recommendation:** Update Section 10 in `AGENTS.md` to reflect current completion state.

---

### 3.3. Missing Error Handling

**Issue (Medium severity):** In `ComplexCUDAPCGPositiveDefiniteSolver::solve()`, the output vector `realX` is pre-filled from `x.real()` / `x.imag()` before calling `realSolver->solve(realX, realRhs)`. However, the underlying `CUDAPCGInternals::solve()` always initialises `d_x` to zero (`cudaMemsetAsync(d_x, 0, ...)`), ignoring any warm-start content passed from `realX`. The `realX` pre-fill at lines 616–617 of `cuda_pcg_solver.cu` is therefore dead code and misleads callers about warm-start support.

**Recommendation:** Either implement warm-start in `CUDAPCGInternals::solve()` by copying `h_x` to `d_x` before the PCG loop, or remove the pre-fill and zero `realX` explicitly. Current behaviour is correct (starts from zero) but the code is deceptive.

---

**Issue (Low severity):** `cudaEventElapsedTime()` return values (lines 434–436 of `cuda_pcg_solver.cu`) are not checked against `CUDA_SUCCESS`. A failed event-elapsed call silently leaves timing fields at 0.0. This does not affect correctness but corrupts benchmark output.

**Recommendation:** Wrap each `cudaEventElapsedTime` with `GC_CUDA_CHECK`.

---

**Issue (Low severity):** If `denom <= 0` breakdown occurs inside the PCG loop (line 390 of `cuda_pcg_solver.cu`), a `std::runtime_error` is thrown. CUDA device state at that point may have pending async operations on `stream`. The throw bypasses the `cudaStreamSynchronize` call that normally appears after the loop. This leaves the stream in a pending state; subsequent destructor calls (`cusparseDestroy`, etc.) are safe because they drain the stream internally, but it is fragile.

**Recommendation:** Add `cudaStreamSynchronize(stream)` before throwing from within the loop body, or use a flag-and-break pattern to exit cleanly.

---

### 3.4. CUDA Resource Leaks

**No confirmed leaks detected.**

All CUDA handles (`cusparseHandle`, `cublasHandle`, `cudaStream_t`, 6 CUDA events), all device buffers (`d_rowOffsets`, `d_colIndices`, `d_values`, `d_diagInv`, `d_x`, `d_b`, `d_r`, `d_z`, `d_p`, `d_q`, `d_spmvBuffer`), and all cuSPARSE descriptors (`matDescr`, `vecPDescr`, `vecQDescr`, `vecXDescr`) are freed in `CUDAPCGInternals::~CUDAPCGInternals()` and `freeBuffers()`. Null guards prevent double-free. The destructor is called via the `unique_ptr` owned by `CUDAPCGPositiveDefiniteSolver`.

**Potential issue (Low severity):** `cudaMallocAsync` is used (lines 256–294) but the corresponding stream is created just before. If `cudaMallocAsync` fails and throws through `GC_CUDA_CHECK`, partial allocations from earlier lines in the same `setMatrix()` call will be freed when `freeBuffers()` is called from the destructor via RAII — only if the pointers were already stored in the struct. Because assignments happen immediately after each `cudaMallocAsync` call, the destructor will see non-null pointers and free them correctly. This is safe.

---

### 3.5. Unnecessary CPU-GPU Transfers

**No unnecessary transfers detected inside `solve()`.**

Per-solve transfers are: one H2D copy of `rhs` (`d_b ← rhs`, $N$ doubles) and one D2H copy of the solution (`h_x ← d_x`, $N$ doubles). All working vectors (`d_r`, `d_z`, `d_p`, `d_q`) remain on device throughout the PCG loop. No norm values are copied device-to-host during the loop; `cublasDnrm2` writes to a host pointer via a cuBLAS stream (followed by `cudaStreamSynchronize`).

**Potential improvement:** The `bNorm` computation (line 323) and `rNorm` computation (line 344) each call `cudaStreamSynchronize`, giving 2 synchronisation barriers before the loop even starts. These could be fused into a single kernel that computes both norms simultaneously to reduce synchronisation cost. This is an optimisation, not a correctness issue.

---

### 3.6. Poor Documentation

**Issue (Low severity):** `cuda_pcg_solver.h` declares `solvePositiveDefiniteCUDA<T>()` wrapped in `#ifdef GC_HAVE_CUDA` (lines 98–102). However, the same header is included unconditionally; callers who `#include` this header without `GC_HAVE_CUDA` will not see this function, which is correct, but the conditional guard creates asymmetry with the class declarations above it (which are always visible). A comment explaining this is absent.

**Issue (Low severity):** `BENCHMARKING.md` has a malformed header at line 2 (`# Linear Solver Benchmark Results: CPU vs GPU` immediately followed by another `#` on line 2 of the original, creating a duplicate `h1`). This is a documentation formatting artifact.

**Issue (Low severity):** `AGENTS.md` Section 10 status is stale (see §3.2 above).

---

### 3.7. Missing Tests

**Missing (Medium):** There is no test that exercises `CUDAPCGPositiveDefiniteSolver<float>`. The `float` instantiation compiles and is used in benchmarks, but is not covered by any GoogleTest case. Given that FP32 is confirmed non-convergent for Poisson, a test asserting that FP32 converges for the heat operator but explicitly verifies that it fails (or warns) for the Poisson operator would prevent silent regression.

**Missing (Medium):** There is no test for `ComplexCUDAPCGPositiveDefiniteSolver` in isolation (independent of `VectorHeatMethodSolver`). A unit test building a small Hermitian positive-definite complex matrix and verifying the $2N \times 2N$ real block expansion would improve confidence.

**Missing (Low):** The robustness test binary (`tests/robustness/test_cad_robustness.cpp`) is not integrated into the main GoogleTest suite — it is a standalone executable in the benchmarks build rather than a registered `add_test()`. This means it is excluded from `ctest` runs.

**Missing (Low):** There is no test verifying that requesting `HeatSolverBackend::CUDA_PCG` when the library was built without `GC_ENABLE_CUDA` throws a `std::runtime_error` with a clear message. This path exists in code but is not exercised.

---

### 3.8. Build Portability Issues

**Issue (Medium):** `CMakeLists.txt` (root) hardcodes `CMAKE_CUDA_ARCHITECTURES = 89` as the default when `GC_ENABLE_CUDA=ON` and no architecture is specified (line 31). `sm_89` is specific to NVIDIA Ada Lovelace (L4, RTX 4090). On Ampere (A100, sm_80) or Hopper (H100, sm_90) or Turing (T4, sm_75), this default will either fail to build or produce sub-optimal code.

**Recommendation:** Change the default to `native` (requires CMake 3.24+) or a multi-architecture list such as `75;80;86;89;90`. The current default is fine for the specific development machine (L4) but will silently produce wrong PTX on other GPUs.

---

**Issue (Low):** The root `CMakeLists.txt` sets `cmake_minimum_required(VERSION 3.14.0)` (line 1) but uses `CUDAToolkit` targets (`CUDA::cusparse`, etc.), which require CMake ≥ 3.17. This is technically a version constraint violation. If someone builds on CMake 3.14–3.16 with CUDA enabled, the `find_package(CUDAToolkit)` call will fail.

**Recommendation:** Raise `cmake_minimum_required` to `3.17` when `GC_ENABLE_CUDA=ON`, or add a version check with a clear error message.

---

**Issue (Low):** `cudaMallocAsync` (used in `setMatrix()`) requires CUDA ≥ 11.2. The project targets CUDA 12.6, so this is fine on the current platform. If someone attempts to build with CUDA 10.x or 11.0, they will get a compile error. There is no version guard.

---

**Issue (Low):** `src/CMakeLists.txt` lists `surface/simple_idt.cpp` twice (lines 42 and 55). This is a pre-existing upstream issue unrelated to the GPU backend, but worth noting.

---

### 3.9. Benchmark Methodology

**Confirmed correct:**
- 3 warmup solves per entry (documented and implemented).
- 10 timed repetitions with mean ± std dev reported.
- CUDA events (`cudaEventRecord` / `cudaEventElapsedTime`) used for GPU timing, not `std::chrono`.
- H2D, solve, and D2H times measured separately and independently.
- Setup (factorization / GPU upload) time measured and reported separately from solve time.
- CPU timing uses `std::chrono::high_resolution_clock`.

**Issue (Low):** The CPU iterative CG solver (Eigen Jacobi-CG) listed in the benchmark table as "CPU Iterative (Eigen Jacobi-CG)" is not the production solver path; it is included only as a comparison point. The relevant CPU baseline for users is "CPU Direct (SimplicialLDLT)". The benchmark table could mislead readers into thinking the library uses CPU Jacobi-CG by default.

**Issue (Low):** The benchmark binary does not pin thread affinity or disable CPU frequency scaling. The AMD EPYC 7742 supports boost from 2.25 GHz to 3.40 GHz; CPU timing variance between runs may be influenced by frequency scaling, particularly for short-duration solves (< 1 ms). Standard deviation values in the table are low ($\pm 0.00$ to $\pm 0.01$ ms), suggesting this was not a problem in practice, but the methodology document does not mention this.

---

### 3.10. Unverified Performance Claims

The following claims are **verified by benchmark data in `BENCHMARKING.md`**:

- ✅ Heat diffusion: GPU is **3.1×** faster at 11,714 vertices, **10.1×** at 46,850, **30.5×** at 187,394.
- ✅ Poisson: GPU is **~0.66×** (slower) at 46,850; **~1.3×** faster at 187,394 (solve-only).
- ✅ PCIe transfer overhead < 2% of total solve time.
- ✅ FP32 fails to converge on shifted Poisson for all mesh sizes tested.
- ✅ GPU factorization/setup time is measured in single-digit milliseconds vs. 23+ seconds for CHOLMOD at 187k vertices.

The following claim in `ROBUSTNESS.md` (Section 6.1) is **NOT consistent with the main benchmark table**:

> **GPU PCG Execution: Setup = 0.36 ms, Solve = 163.70 ms, Total = 353.97 ms (698 iters)**
> **CPU Direct Cholesky: Total = 23,970.50 ms → Speedup: 67.7×**

These numbers differ from `BENCHMARKING.md` for the same 187k mesh:
- `BENCHMARKING.md`: GPU solve-only = **20.58 ms** (91 iters, heat diffusion), **471.22 ms** (2000 iters, Poisson).
- `ROBUSTNESS.md`: GPU solve = 163.70 ms (698 iters).

The discrepancy suggests the robustness benchmark was run on a *combined* heat + Poisson solve or on a different operator configuration than the main benchmark table. The "67.7×" claim in `ROBUSTNESS.md` cannot be matched to any single row in `BENCHMARKING.md`.

> [!WARNING]
> Do not quote the "67.7×" or "67.9×" speedup figure from `ROBUSTNESS.md` in publications or presentations. Use only the operator-specific figures from `BENCHMARKING.md`, which report 30.5× for heat diffusion and 1.3× for Poisson at 187k vertices.

---

## 4. Known Limitations

| Limitation | Details |
|---|---|
| **Iterative solver on ill-conditioned Poisson** | Jacobi-PCG requires up to 2,000 iterations on $L + 10^{-6}I$ for meshes with 50k+ vertices. GPU is not faster than CPU Cholesky until >100k vertices for this operator. |
| **No warm-start** | Every `solve()` call initialises $x_0 = 0$. For repeated solves with slowly-changing RHS, warm-starting from the previous solution could halve iteration counts. |
| **FP32 unusable for Poisson** | FP32 fails to converge on $L + 10^{-6}I$ for all tested mesh sizes. Do not enable FP32 for the Poisson system. |
| **Affine logmap on CPU only** | `LogMapStrategy::AffineLocal` and `AffineAdaptive` use `SquareSolver<double>` unconditionally. These solvers involve indefinite/asymmetric systems that are not compatible with PCG. |
| **No GPU gradient/divergence kernels** | The heat method geometry steps (gradient per face, divergence per vertex) are computed on CPU with a D2H / H2D round-trip between the two GPU solves. This adds overhead proportional to mesh size. |
| **Single-GPU only** | No multi-GPU sharding. |
| **CUDA 11.2+ required** | `cudaMallocAsync` requires CUDA ≥ 11.2. |
| **sm_89 hardcoded default** | Default CUDA architecture targets Ada Lovelace only. Must set `CMAKE_CUDA_ARCHITECTURES` manually for other GPUs. |
| **Robustness tests not in ctest** | `test_cad_robustness` is a standalone binary, not registered with `add_test()`. |
| **No install rules for CUDA targets** | `CUDAToolkit` imported targets (`CUDA::cusparse` etc.) are linked `PRIVATE`, so they do not propagate to installed export sets. `cmake --install` may produce a broken package on CUDA-enabled builds if downstream consumers link `geometry-central` as a CMake import. |

---

## 5. Reproducibility Instructions

### 5.1. Prerequisites

```
NVIDIA GPU with compute capability ≥ 7.5 (Turing or newer recommended for sm_75+)
CUDA Toolkit 12.x  (12.6 tested)
CMake ≥ 3.17
GCC ≥ 9 or Clang ≥ 10
Eigen 3.3+
SuiteSparse (optional, for CHOLMOD CPU baseline)
```

### 5.2. Build the Library with CUDA

```bash
cd /DATA/suraj/m1/geometry/geometry-central
cmake -B build -S . \
  -DGC_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Replace `89` with your GPU's compute capability (`75` for T4, `80` for A100, `86` for RTX 3090, `90` for H100).

### 5.3. Build and Run Tests

```bash
cmake -B test/build -S test \
  -DGC_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build test/build -j$(nproc)
cd test/build && ctest --output-on-failure
```

Expected: **209/209 tests pass** (includes 8 CUDA-specific cases under `HeatMethodSuite`).

### 5.4. Build the Standalone Solver and Run Its Tests

```bash
cd /DATA/suraj/m1/geometry/geometry-central/gpu_solver
cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
cd build && ctest --output-on-failure
```

Expected: **13/13 tests pass**.

### 5.5. Run Benchmarks

```bash
cd /DATA/suraj/m1/geometry/geometry-central
cmake -B benchmarks/build -S benchmarks \
  -DGC_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build benchmarks/build -j$(nproc)

# Heat method benchmark
./benchmarks/build/test_heat_method_gpu

# Vector heat benchmark
./benchmarks/build/test_vector_heat_gpu
```

### 5.6. Run Robustness Tests

```bash
./benchmarks/build/test_cad_robustness
```

Expected: exits 0. Each of the 7 failure categories is exercised and its behaviour reported to stdout.

---

## 6. Suggested Future Improvements

### 6.1. High Priority

1. **GPU gradient and divergence kernels** — Eliminate the CPU round-trip between the two heat method solves. Face gradient and vertex divergence are embarrassingly parallel. This would remove the last CPU-GPU transfer within the heat method pipeline and potentially add another 1.5–3× speedup for large meshes.

2. **Warm-start support** — Copy an initial `x0` to `d_x` instead of always zeroing it. This is a one-line change to `CUDAPCGInternals::solve()` and could halve iteration counts for repeated solves with slowly-varying RHS (e.g., animation frames).

3. **Register robustness test with ctest** — Add `add_test(NAME CadRobustness COMMAND test_cad_robustness)` to prevent silent regressions.

4. **Fix `cmake_minimum_required` for CUDA** — Raise from `3.14` to `3.17` (minimum for `CUDAToolkit` import targets).

5. **Multi-architecture CUDA default** — Replace `CMAKE_CUDA_ARCHITECTURES=89` default with `native` (CMake 3.24+) or `75;80;86;89;90`.

### 6.2. Medium Priority

6. **Incomplete Cholesky or ILU(0) preconditioner** — Jacobi (diagonal) scaling is the weakest useful preconditioner. For the ill-conditioned shifted Poisson operator, an incomplete Cholesky preconditioner (available in cuSPARSE via `cusparseCreateCsric02Info`) would reduce iteration counts dramatically, potentially making GPU PCG competitive with CPU Cholesky at much smaller mesh sizes.

7. **Unit test for `CUDAPCGPositiveDefiniteSolver<float>`** — Confirm heat diffusion convergence and Poisson non-convergence in an automated test.

8. **Unit test for `ComplexCUDAPCGPositiveDefiniteSolver`** — Independent of `VectorHeatMethodSolver`; build a small $4\times4$ HPD complex matrix and verify the $2N \times 2N$ real solve recovers the correct complex solution.

9. **Fix `BENCHMARKING.md` duplicate table** — Remove the plain-text table (lines 15–96) and retain the LaTeX-formatted version only.

10. **Fix stale `AGENTS.md` Section 10** — Update status to reflect completion of all four phases.

### 6.3. Low Priority

11. **Install rules for CUDA** — Add `cmake_language(DEFER ...)` or an install-time `find_dependency(CUDAToolkit)` shim in `GeometryCentralConfig.cmake` so that installed packages with CUDA enabled are usable downstream.

12. **`#pragma once` guard for `solvePositiveDefiniteCUDA`** — Add a comment explaining why `solvePositiveDefiniteCUDA` is conditionally declared within the always-included header.

13. **Breakdown exception with stream sync** — Add `cudaStreamSynchronize(stream)` before throwing from within the PCG loop to prevent leaving the CUDA stream in an inconsistent state.

14. **`cudaEventElapsedTime` error checking** — Wrap timing calls with `GC_CUDA_CHECK`.

15. **Signed heat method GPU support** — `src/surface/signed_heat_method.cpp` solves similar SPD systems but was not extended to support `CUDA_PCG`. A future phase could add `SignedHeatSolverBackend`.

---

## 7. Summary Assessment

| Category | Status |
|---|---|
| Core PCG algorithm correctness | ✅ Verified by tests and CPU/GPU agreement checks |
| CUDA resource management (RAII) | ✅ Clean; no leaks detected |
| CPU build not affected by GPU code | ✅ All `#ifdef GC_HAVE_CUDA` guards in place |
| Existing test suite compatibility | ✅ 209/209 upstream tests pass |
| Benchmark methodology | ✅ Sound (CUDA events, warmup, repetitions) |
| Performance claims | ✅ Supported by data, with one inconsistency flagged in ROBUSTNESS.md |
| Build portability | ⚠️ Hardcoded `sm_89`; `cmake_minimum_required` too low for CUDA |
| Test coverage | ⚠️ Missing FP32 unit test; `ComplexCUDAPC`G unit test; robustness not in ctest |
| Documentation | ⚠️ Duplicate benchmark table; stale AGENTS status; one unexplained RAII asymmetry |
| Error handling | ⚠️ Three minor issues (breakdown throw ordering; timing unchecked; dead warm-start code) |

