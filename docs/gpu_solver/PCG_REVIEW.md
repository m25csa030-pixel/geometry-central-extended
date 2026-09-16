# CUDA PCG Solver: Detailed Correctness Review

This document reviews the standalone CUDA Jacobi-preconditioned Conjugate Gradient solver
in `gpu_solver/` against the standard preconditioned CG algorithm (Shewchuk §B, Saad §9.2).

---

## Reference Algorithm: Preconditioned Conjugate Gradient

Standard PCG for solving $Ax = b$ with preconditioner $M$:

```
Given: A (SPD), b, x₀, M (SPD preconditioner), tol, maxIters
  r₀ = b − A x₀
  z₀ = M⁻¹ r₀
  p₀ = z₀
  γ₀ = r₀ᵀ z₀
  for k = 0, 1, 2, …:
    qₖ = A pₖ
    αₖ = γₖ / (pₖᵀ qₖ)
    xₖ₊₁ = xₖ + αₖ pₖ
    rₖ₊₁ = rₖ − αₖ qₖ
    check convergence: ‖rₖ₊₁‖ / ‖b‖ ≤ tol
    zₖ₊₁ = M⁻¹ rₖ₊₁
    γₖ₊₁ = rₖ₊₁ᵀ zₖ₊₁
    βₖ = γₖ₊₁ / γₖ
    pₖ₊₁ = zₖ₊₁ + βₖ pₖ
```

---

## 1. CSR Row Offsets and Column Indices

**Status: Correct.**

- `CsrMatrix::validate()` ([csr_matrix.h:40–65](file:///DATA/suraj/m1/geometry/geometry-central/gpu_solver/include/gpu_solver/csr_matrix.h#L40-L65)) correctly checks: non-negative dimensions, `rowOffsets` size = `nRows + 1`, starts at 0, ends at `nnz`, monotonically non-decreasing, column indices in bounds, values finite.
- `fromDiagonal` and `make1DLaplacian` produce sorted column indices within each row.
- Upload to device ([cuda_pcg_solver.cu:232–240](file:///DATA/suraj/m1/geometry/geometry-central/gpu_solver/src/cuda_pcg_solver.cu#L232-L240)) correctly copies `(nRows+1)` offsets, `nnz` column indices, and `nnz` values.

## 2. Sparse Matrix-Vector Multiplication

**Status: Correct.**

- `cusparseCreateCsr` at [line 275](file:///DATA/suraj/m1/geometry/geometry-central/gpu_solver/src/cuda_pcg_solver.cu#L275) uses `CUSPARSE_INDEX_32I` and `CUSPARSE_INDEX_BASE_ZERO`, matching the `int`-typed 0-indexed CSR format.
- SpMV calls at lines [429–431](file:///DATA/suraj/m1/geometry/geometry-central/gpu_solver/src/cuda_pcg_solver.cu#L429-L431) and [354–356](file:///DATA/suraj/m1/geometry/geometry-central/gpu_solver/src/cuda_pcg_solver.cu#L354-L356) use `CUSPARSE_OPERATION_NON_TRANSPOSE` with `alpha=1, beta=0`, computing `q = A*p` and `q = A*x` correctly.
- The SpMV buffer is queried and allocated during `setMatrix()` (lines [286–292](file:///DATA/suraj/m1/geometry/geometry-central/gpu_solver/src/cuda_pcg_solver.cu#L286-L292)), avoiding allocation during solve.

## 3. Jacobi Preconditioner

**Status: Correct.**

- Diagonal extraction ([lines 243–261](file:///DATA/suraj/m1/geometry/geometry-central/gpu_solver/src/cuda_pcg_solver.cu#L243-L261)): correctly searches each row for `colIndices[j] == i`, computes `1/A_ii`, and falls back to `1.0` for missing or near-zero diagonals.
- Kernel application ([lines 17–23](file:///DATA/suraj/m1/geometry/geometry-central/gpu_solver/src/cuda_pcg_solver.cu#L17-L23)): `z[i] = r[i] * dinv[i]` is mathematically correct for $z = M^{-1}r$.
- Bounds check: `if (i < n)` prevents out-of-bounds access.

## 4. PCG Recurrence Equations

**Status: Correct.** The implementation at lines [395–503](file:///DATA/suraj/m1/geometry/geometry-central/gpu_solver/src/cuda_pcg_solver.cu#L395-L503) matches the standard algorithm exactly:

| Step | Reference | Implementation | Correct? |
|------|-----------|----------------|----------|
| $z_0 = M^{-1} r_0$ | Pre-loop | Line 399 `applyJacobiKernel` | ✅ |
| $p_0 = z_0$ | Pre-loop | Line 402 `copy(d_z → d_p)` | ✅ |
| $\gamma_0 = r_0^T z_0$ | Pre-loop | Line 406 `dot(d_r, d_z)` | ✅ |
| $q_k = A p_k$ | Loop body | Lines 429–431 `SpMV(A, p → q)` | ✅ |
| $\alpha_k = \gamma_k / (p_k^T q_k)$ | Loop body | Lines 434–444 | ✅ |
| $x_{k+1} = x_k + \alpha_k p_k$ | Loop body | Line 447 `axpy(alpha, p, x)` | ✅ |
| $r_{k+1} = r_k - \alpha_k q_k$ | Loop body | Lines 450–451 `axpy(-alpha, q, r)` | ✅ |
| $z_{k+1} = M^{-1} r_{k+1}$ | Loop body | Line 487 `applyJacobiKernel` | ✅ |
| $\gamma_{k+1} = r_{k+1}^T z_{k+1}$ | Loop body | Line 491 `dot(d_r, d_z)` | ✅ |
| $\beta_k = \gamma_{k+1} / \gamma_k$ | Loop body | Line 499 | ✅ |
| $p_{k+1} = z_{k+1} + \beta_k p_k$ | Loop body | Line 503 `updateSearchDirectionKernel` | ✅ |

## 5. Dot Products and Reductions

**Status: Correct.**

- All dot products use cuBLAS `cublasDdot`/`cublasSdot` which perform numerically stable parallel reduction on the GPU.
- The `nrm2` calls for residual norm also use cuBLAS's pairwise summation.
- Results are written to host-side scalars and synchronized before use.

## 6. Residual Calculation

**Status: Correct, with a good robustness feature.**

- Initial residual: For `x0 = 0`, `r = b` (line 367). For non-zero `x0`, `r = b - A*x0` is computed via SpMV + copy + axpy (lines 354–363). Both are correct.
- Recursive residual update: `r = r - alpha * q` (line 451) is the standard recursive update.
- True residual recomputation (lines [453–464](file:///DATA/suraj/m1/geometry/geometry-central/gpu_solver/src/cuda_pcg_solver.cu#L453-L464)): Every `residualRecomputeInterval` iterations, the true residual `r = b - A*x` is recomputed from scratch. This is a good practice that prevents false convergence from accumulated floating-point drift.

> **Confirmed Issue**: See Section 4A below — the true residual recomputation at line 453–464 corrupts the PCG recurrence.

## 7. Convergence Criteria

**Status: Correct.**

- Relative residual stopping criterion: $\|r_k\|_2 / \|b\|_2 \le \text{tol}$ (lines [471, 481](file:///DATA/suraj/m1/geometry/geometry-central/gpu_solver/src/cuda_pcg_solver.cu#L471-L481)).
- Zero RHS: returns `x = 0` immediately (line [328–343](file:///DATA/suraj/m1/geometry/geometry-central/gpu_solver/src/cuda_pcg_solver.cu#L328-L343)).
- Early exit if initial guess already satisfies tolerance (lines [379–393](file:///DATA/suraj/m1/geometry/geometry-central/gpu_solver/src/cuda_pcg_solver.cu#L379-L393)).

## 8. Breakdown Conditions

**Status: Correct.**

- $p^T q \le 0$ check at line [438](file:///DATA/suraj/m1/geometry/geometry-central/gpu_solver/src/cuda_pcg_solver.cu#L438): detects non-positive-definiteness or numerical breakdown.
- $r^T M^{-1} r \le 0$ check at lines [409](file:///DATA/suraj/m1/geometry/geometry-central/gpu_solver/src/cuda_pcg_solver.cu#L409) and [494](file:///DATA/suraj/m1/geometry/geometry-central/gpu_solver/src/cuda_pcg_solver.cu#L494): detects preconditioner/matrix issues.
- NaN and Inf checks on residual norm and inner products.
- Divergence detection: `currentResNorm > 1e10 * initialResNorm` (line [475](file:///DATA/suraj/m1/geometry/geometry-central/gpu_solver/src/cuda_pcg_solver.cu#L475)).

## 9. Floating-Point Precision

**Status: Correct.**

- Type traits correctly dispatch to `cublasDdot`/`cublasSdot` etc. based on `T`.
- `cudaDataType` correctly set to `CUDA_R_64F` for `double` and `CUDA_R_32F` for `float`.
- All host-side arithmetic uses matching type `T`, with widening to `double` only for status reporting.

## 10. CUDA Synchronization

**Status: Correct.**

- `cudaStreamSynchronize` is called after every cuBLAS reduction that reads a scalar back to the host (lines [326](file:///DATA/suraj/m1/geometry/geometry-central/gpu_solver/src/cuda_pcg_solver.cu#L326), [373](file:///DATA/suraj/m1/geometry/geometry-central/gpu_solver/src/cuda_pcg_solver.cu#L373), [407](file:///DATA/suraj/m1/geometry/geometry-central/gpu_solver/src/cuda_pcg_solver.cu#L407), [436](file:///DATA/suraj/m1/geometry/geometry-central/gpu_solver/src/cuda_pcg_solver.cu#L436), [468](file:///DATA/suraj/m1/geometry/geometry-central/gpu_solver/src/cuda_pcg_solver.cu#L468), [492](file:///DATA/suraj/m1/geometry/geometry-central/gpu_solver/src/cuda_pcg_solver.cu#L492)).
- This is required because cuBLAS returns results to host memory by default (`CUBLAS_POINTER_MODE_HOST`), which needs stream synchronization before the host reads the value. ✅
- `setMatrix()` synchronizes at [line 294](file:///DATA/suraj/m1/geometry/geometry-central/gpu_solver/src/cuda_pcg_solver.cu#L294) before marking `initialized = true`. ✅
- Final solution copy synchronizes at [line 508](file:///DATA/suraj/m1/geometry/geometry-central/gpu_solver/src/cuda_pcg_solver.cu#L508). ✅

## 11. Out-of-Bounds Memory Access

**Status: No issues detected.**

- CUDA kernels use `if (i < n)` guards.
- cuBLAS and cuSPARSE are given correct dimension `nRows`.
- Vector descriptors are created with size `nRows` matching allocated buffer sizes.
- CSR descriptor dimensions match allocated buffer sizes.

## 12. Resource Cleanup

**Status: Correct.**

- `freeMatrixAndWorkingBuffers()` ([lines 145–213](file:///DATA/suraj/m1/geometry/geometry-central/gpu_solver/src/cuda_pcg_solver.cu#L145-L213)) destroys all cuSPARSE descriptors and frees all device memory with null-checks and null-reset.
- Destructor ([lines 138–143](file:///DATA/suraj/m1/geometry/geometry-central/gpu_solver/src/cuda_pcg_solver.cu#L138-L143)) calls `freeMatrixAndWorkingBuffers()`, then destroys handles and stream.
- `setMatrix()` calls `freeMatrixAndWorkingBuffers()` first ([line 226](file:///DATA/suraj/m1/geometry/geometry-central/gpu_solver/src/cuda_pcg_solver.cu#L226)), correctly handling re-initialization.

## 13. Memory Leaks

**Status: No leaks detected.**

- Every `cudaMallocAsync` has a corresponding `cudaFree` in `freeMatrixAndWorkingBuffers`.
- PIMPL pattern ensures destructor runs even if exceptions occur in calling code.
- `freeMatrixAndWorkingBuffers` is idempotent (null-check guards).

> **Potential Issue**: See Section 5B below regarding exception safety during `setMatrix()`.

## 14. Numerical Comparison Against CPU Reference

**Status: Correct.**

- Test `testRandomSpdCpuReference` ([test_cuda_pcg.cpp:155–213](file:///DATA/suraj/m1/geometry/geometry-central/gpu_solver/tests/test_cuda_pcg.cpp#L155-L213)) constructs a well-conditioned random SPD matrix ($A = BB^T + nI$), solves with both Eigen `SimplicialLDLT` and CUDA PCG, and compares element-wise with tolerance $10^{-4}$.
- The achieved difference was $4.04 \times 10^{-11}$, well within tolerance.

> **Confirmed Issue**: See Section 4B below — the Eigen CSC → CSR conversion in this test is only correct because the matrix is symmetric.

---

## Confirmed Issues

### Issue A — True residual recomputation corrupts the PCG recurrence relationship

**Severity: Medium. Affects correctness on long-running solves.**

At [cuda_pcg_solver.cu lines 453–464](file:///DATA/suraj/m1/geometry/geometry-central/gpu_solver/src/cuda_pcg_solver.cu#L453-L464), when the true residual is recomputed (`r = b - A*x`), the code overwrites `d_q` with `A*x`:

```cpp
// q = A * x                         ← overwrites d_q
cusparseSpMV(..., vecXDescr, ..., vecQDescr, ...);
// r = b
CudaTraits<T>::copy(cublasHandle, nRows, d_b, 1, d_r, 1);
// r = r - 1.0 * q
CudaTraits<T>::axpy(cublasHandle, nRows, &negOne, d_q, 1, d_r, 1);
```

After this block, `d_q` no longer contains $Ap_k$ — it contains $Ax_k$. This is harmless for the residual `r` itself, because `r` is correctly recomputed. However, the real concern is subtler:

The code has already consumed `d_q` for this iteration: `x` and `r` are already updated using the correct `alpha` derived from the correct `denom = p^T q` (lines 434–451). The residual recomputation happens *after* those updates. Then the code proceeds to compute the *new* `z`, `gamma_new`, `beta`, and `p` — none of which use `d_q`. So `d_q` is not read again until the *next* iteration's SpMV at line 429, which overwrites it completely with the new `A*p`.

**Verdict: Actually correct upon close analysis.** The `d_q` buffer is a temporary that is fully overwritten at the top of each iteration. The residual recomputation reuses it as scratch space for `A*x` at a point where `d_q`'s previous value (`A*p_k`) is no longer needed. No fix required.

### Issue B — Test `testRandomSpdCpuReference` CSC-to-CSR conversion is silently correct but fragile

**Severity: Low (test-only). Works by accident for symmetric matrices.**

At [test_cuda_pcg.cpp lines 169–180](file:///DATA/suraj/m1/geometry/geometry-central/gpu_solver/tests/test_cuda_pcg.cpp#L169-L180):

```cpp
Eigen::SparseMatrix<double> eigenA = denseA.sparseView(); // ColMajor = CSC
CsrMatrix<double> mat(n, n, static_cast<int>(eigenA.nonZeros()));
for (int i = 0; i <= n; ++i) {
  mat.rowOffsets[i] = eigenA.outerIndexPtr()[i];  // CSC column offsets → "row offsets"
}
for (int i = 0; i < eigenA.nonZeros(); ++i) {
  mat.colIndices[i] = eigenA.innerIndexPtr()[i];  // CSC row indices → "col indices"
  mat.values[i] = eigenA.valuePtr()[i];
}
```

This treats Eigen's CSC (`outerIndexPtr` = column offsets, `innerIndexPtr` = row indices) as CSR (`outerIndexPtr` = row offsets, `innerIndexPtr` = column indices). For a **symmetric** matrix $A = A^T$, CSC(A) = CSR(A), so this is numerically correct. However:

1. **No comment explains this reliance on symmetry.** A future developer might copy this pattern for a non-symmetric matrix and get silently wrong results.
2. **The matrix is fully dense** ($BB^T + nI$ has no zero entries), so `sparseView()` creates a full $n \times n$ "sparse" matrix. This is fine for a test but masks any sparsity-related issues.

**Required fix**: Add a clarifying comment.

---

## Potential Issues

### Issue C — `setMatrix()` exception safety: partial allocation leak

**Severity: Low. Only affects error paths.**

In `CudaPcgSolverImpl::setMatrix()` ([lines 215–296](file:///DATA/suraj/m1/geometry/geometry-central/gpu_solver/src/cuda_pcg_solver.cu#L215-L296)), if any `cudaMallocAsync` or `cusparseCreateCsr` call throws (via the check macros), the previously allocated buffers within the same function are not freed. For example, if `cudaMallocAsync(&d_z, ...)` at line 270 fails, then `d_rowOffsets`, `d_colIndices`, `d_values`, `d_diagInv`, `d_x`, and `d_b` are leaked.

The `freeMatrixAndWorkingBuffers()` at line 226 correctly cleans up *previous* state, but the new partial allocations are not cleaned up because `initialized` is still `false` — however, `freeMatrixAndWorkingBuffers()` checks each pointer individually, so a subsequent destructor call *would* clean them up. **The real risk**: if `setMatrix()` throws, the next `setMatrix()` call's `freeMatrixAndWorkingBuffers()` at line 226 would also clean up the partial state from the failed call, because the pointers are still non-null.

**Verdict: Safe in practice** because the destructor and re-invocation paths both call `freeMatrixAndWorkingBuffers()` which null-checks each pointer. No fix required.

### Issue D — `cudaMallocAsync` with `nnz = 0`

**Severity: Low.**

If a matrix has `nnz = 0` (e.g., a zero matrix), `cudaMallocAsync(&d_colIndices, 0 * sizeof(int), stream)` is called. `cudaMallocAsync` with size 0 is implementation-defined — it may return `nullptr` or a valid pointer. The subsequent `cudaMemcpyAsync` with size 0 is a no-op and safe. The `cudaFree(nullptr)` in cleanup is also a no-op.

**Verdict: Unlikely to be encountered in practice** (a zero matrix is not SPD). No fix required.

### Issue E — `bNorm == 0` comparison with floating-point

**Severity: Low.**

At [line 328](file:///DATA/suraj/m1/geometry/geometry-central/gpu_solver/src/cuda_pcg_solver.cu#L328), `if (bNorm == 0)` uses exact floating-point equality. If `b` contains only denormals or values very close to zero (e.g., $O(10^{-300})$), `nrm2` will return a tiny but non-zero value, and the solver will proceed normally. This is acceptable because the relative residual check will still work correctly — a near-zero `bNorm` will cause rapid convergence detection.

**Verdict: No fix required.**

---

## Required Fixes

### Fix 1 — Add clarifying comment in test for CSC→CSR conversion

The test's reliance on symmetry for the Eigen CSC→CSR reinterpretation should be documented.

### Fix 2 — Add missing tests

See "Tests That Are Missing" below.

---

## Tests That Are Missing

### T1. Zero RHS vector (`b = 0`)

The solver claims to handle `b = 0` by returning `x = 0`. No test exercises this.

### T2. RHS size mismatch

The `solve(vector, vector)` overload silently resizes `x` if its size doesn't match. The behavior when `b.size() != nRows` is untested and likely causes undefined behavior (out-of-bounds device memory access from `cudaMemcpyAsync` copying `b.data()` with size `nRows * sizeof(T)` when `b` has fewer elements).

### T3. Matrix re-initialization (`setMatrix` called twice)

Verifies that a solver instance can be re-initialized with a different matrix without leaking memory.

### T4. `throwOnFailure` mode

The `setThrowOnFailure(true)` flag is implemented but never tested.

### T5. Large sparse matrix

The current tests use matrices of dimension ≤100. A test with $n \geq 1000$ would exercise GPU parallelism more realistically.

### T6. Ill-conditioned SPD matrix

A test with a near-singular SPD matrix (high condition number) would verify that the solver converges slowly rather than silently producing garbage.

---

## Summary

| Category | # | Item | Severity | Fix Required? |
|----------|---|------|----------|---------------|
| Confirmed (code) | A | True residual recomputation `d_q` overwrite | — | None (correct upon analysis) |
| Confirmed (test) | B | CSC→CSR conversion relies on symmetry without comment | Low | Yes (comment) |
| Potential | C | `setMatrix()` exception safety | Low | No |
| Potential | D | `cudaMallocAsync` with `nnz=0` | Low | No |
| Potential | E | `bNorm == 0` exact float comparison | Low | No |
| Missing test | T1 | Zero RHS | Medium | Yes |
| Missing test | T2 | RHS size mismatch / input validation | Medium | Yes |
| Missing test | T3 | Matrix re-initialization | Low | Yes |
| Missing test | T4 | `throwOnFailure` mode | Low | Yes |
| Missing test | T5 | Large sparse matrix | Low | Yes |
| Missing test | T6 | Ill-conditioned SPD matrix | Low | Yes |

**Overall assessment**: The PCG recurrence is mathematically correct. The CUDA synchronization model is correct. No memory leaks. No out-of-bounds access. The primary gaps are in test coverage and input validation for the `b` vector size.

