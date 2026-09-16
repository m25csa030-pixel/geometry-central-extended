# Robustness Analysis and Testing Workflow for CAD-Derived Meshes

This document establishes a rigorous robustness testing workflow for CAD-derived and topologically imperfect meshes in `geometry-central`, analyzing failure modes across CPU and GPU backends.

---

## 1. Robustness Testing Framework Overview

CAD-derived surface meshes frequently violate idealized manifold and geometric assumptions. Rather than treating all invalid meshes as fatal bugs or silently ignoring errors, `geometry-central`'s architecture decomposes processing into five distinct pipeline stages:

1. **Stage 1: Ingestion & Topological Validation**: Parsing file format (OBJ, PLY, STL) and constructing halfedge connectivity (`ManifoldSurfaceMesh` vs. general `SurfaceMesh`).
2. **Stage 2: Geometric Preprocessing**: Computing edge lengths, corner angles, and face areas (`VertexPositionGeometry`).
3. **Stage 3: Discrete Operator Assembly**: Constructing cotangent Laplacian $L$, vertex lumped mass matrix $M$, and short-time heat flow operators $A = M + t L$.
4. **Stage 4: Linear Solver Execution**: Factorization and iterative solve on CPU (`PositiveDefiniteSolver`) vs. GPU (`CUDAPCGPositiveDefiniteSolver`).
5. **Stage 5: Downstream Geodesic Workflows**: Executing the Heat Method (`HeatMethodDistanceSolver`) in standard mode vs. intrinsic robust mode (`useRobustLaplacian = true`).

### Reproduction Command
All tests can be executed via the dedicated robustness binary:
```bash
./benchmarks/build/test_cad_robustness
```

---

## 2. Detailed Failure Modes and Analysis by Category

### Category 1: Non-Manifold Meshes

#### Failure Case 1.1: Non-Manifold Edge (Shared by $\ge 3$ Faces)
- **Input Mesh**: [`tests/robustness/meshes/nonmanifold_edge.obj`](file:///DATA/suraj/m1/geometry/geometry-central/tests/robustness/meshes/nonmanifold_edge.obj) (3 triangles meeting along edge $v_0 - v_1$).
- **Reproduction**:
  ```cpp
  auto [mesh, geom] = readManifoldSurfaceMesh("tests/robustness/meshes/nonmanifold_edge.obj");
  ```
- **Failure Stage**: **Stage 1 (Loading / Topological Validation)**.
- **Error Message**:
  ```
  GC_SAFETY_ASSERT FAILURE from src/surface/manifold_surface_mesh.cpp:109 - duplicate edge in list 0 -- 1
  ```
- **CPU Behavior**: Aborts with `std::runtime_error` during manifold construction.
- **GPU Behavior**: Identical (loading is host-side).
- **Root Cause**: `ManifoldSurfaceMesh` enforces that each undirected edge has at most two incident faces (one halfedge and its twin). Edges with $\ge 3$ incident faces violate the manifold 2-manifold disk-neighborhood condition.
- **Proposed Fix & Mitigation**:
  1. Load using `readSurfaceMesh()` (instantiating general `SurfaceMesh`), which allows general delta complexes.
  2. In `HeatMethodDistanceSolver`, specify `useRobustLaplacian = true`. This invokes `buildIntrinsicTuftedCover()`, doubling non-manifold edges to create an intrinsically manifold Delaunay cover, which successfully solves on both CPU and GPU.

---

#### Failure Case 1.2: Non-Manifold Pinch Vertex (Hourglass Topology)
- **Input Mesh**: [`tests/robustness/meshes/nonmanifold_pinch_vertex.obj`](file:///DATA/suraj/m1/geometry/geometry-central/tests/robustness/meshes/nonmanifold_pinch_vertex.obj) (Two cones meeting at a single vertex $v_0$).
- **Reproduction**:
  ```cpp
  auto [mesh, geom] = readManifoldSurfaceMesh("tests/robustness/meshes/nonmanifold_pinch_vertex.obj");
  ```
- **Failure Stage**: **Stage 1 (Loading / Topological Validation)**.
- **Error Message**:
  ```
  GC_SAFETY_ASSERT FAILURE from src/surface/manifold_surface_mesh.cpp:158 - vertex 0 appears in more than one boundary loop
  ```
- **CPU Behavior**: Throws `std::runtime_error` during boundary loop resolution.
- **GPU Behavior**: Identical.
- **Root Cause**: The link of vertex $v_0$ is two disconnected circles/wedges rather than a single topological circle or open disk.
- **Proposed Fix**: Use `readSurfaceMesh()` + `useRobustLaplacian = true`.

---

### Category 2: Degenerate Triangles

#### Failure Case 2.1: Self-Edge in Face List
- **Input Mesh**: [`tests/robustness/meshes/degenerate_self_edge.obj`](file:///DATA/suraj/m1/geometry/geometry-central/tests/robustness/meshes/degenerate_self_edge.obj) (Face defined as `f 1 1 2`).
- **Reproduction**:
  ```cpp
  auto [mesh, geom] = readManifoldSurfaceMesh("tests/robustness/meshes/degenerate_self_edge.obj");
  ```
- **Failure Stage**: **Stage 1 (Mesh Construction)**.
- **Error Message**:
  ```
  GC_SAFETY_ASSERT FAILURE from src/surface/manifold_surface_mesh.cpp:107 - self-edge in face list 0 -- 0
  ```
- **CPU Behavior**: Fails at line 107 of `manifold_surface_mesh.cpp`. When forced through general `SurfaceMesh`, fails at Stage 3:
  `checkFinite() failure: Non-finite matrix entry [0,0] = -nan`.
- **GPU Behavior**: Identical host-side validation.
- **Root Cause**: Two adjacent vertices in a polygon index list are identical, producing an edge of zero length and zero face area.
- **Proposed Fix**:
  - Filter repeated consecutive indices during mesh ingestion in `PolygonSoupMesh`.
  - When processed with `useRobustLaplacian = true`, intrinsic mollification ($10^{-5}$) assigns non-zero synthetic edge lengths, resolving the singularity and enabling valid CPU and GPU execution.

---

### Category 3: Duplicate Vertices (Geometric Coincidence)

#### Failure Case 3.1: Coincident Seam Vertices
- **Input Mesh**: [`tests/robustness/meshes/duplicate_coincident_vertices.obj`](file:///DATA/suraj/m1/geometry/geometry-central/tests/robustness/meshes/duplicate_coincident_vertices.obj) (Vertices $v_0, v_1$ have coordinates `(0,0,0)` but different indices).
- **Reproduction**:
  ```cpp
  auto [mesh, geom] = readManifoldSurfaceMesh("tests/robustness/meshes/duplicate_coincident_vertices.obj");
  ```
- **Failure Stage**: Passes Stage 1 (topologically distinct boundaries).
- **CPU Behavior**: Factors and solves without error.
- **GPU Behavior**: Solves identically to CPU.
- **Root Cause**: Topological boundaries prevent geometric coincidence from causing topological singularities.
- **Recommendation**: In CAD workflows where coincident vertices represent split seams, call `PolygonSoupMesh::stripUnusedVertices()` and spatial hashing / welding before halfedge creation if a watertight surface is required.

---

### Category 4: Zero-Area Faces (Collinear Vertices)

#### Failure Case 4.1: Distinct Collinear Vertices in Triangle
- **Input Mesh**: [`tests/robustness/meshes/zero_area_collinear.obj`](file:///DATA/suraj/m1/geometry/geometry-central/tests/robustness/meshes/zero_area_collinear.obj) (Triangle with $v_0=(0,0,0), v_1=(1,0,0), v_2=(2,0,0)$).
- **Reproduction**:
  ```cpp
  auto [mesh, geom] = readSurfaceMesh("tests/robustness/meshes/zero_area_collinear.obj");
  geom->requireCotanLaplacian();
  ```
- **Failure Stage**: **Stage 3 (Matrix Assembly)**.
- **Error Message**:
  ```
  Matrix contains NaN Inf
  checkFinite() failure: Non-finite matrix entry [0,0] = -nan
  ```
- **CPU Behavior**: Standard solver fails during matrix validation (`checkFinite()`).
- **GPU Behavior**: If bypassed, `CUDAPCGPositiveDefiniteSolver` catches non-SPD/NaN inner product and throws:
  `CUDAPCGPositiveDefiniteSolver: Initial inner product <= 0 (matrix is not SPD)`.
- **Root Cause**: For a collinear triangle, height $h \to 0$ and corner angles approach $0^\circ$ or $180^\circ$. The cotangent weight $\cot \theta = \cos \theta / \sin \theta \to \infty$. Cross product area $\frac{1}{2} \|e_1 \times e_2\| = 0$, dividing by zero in cotangent formulas.
- **Proposed Fix**:
  - **Healed by Robust Mode**: When `useRobustLaplacian = true` is enabled in `HeatMethodDistanceSolver`, intrinsic mollification widens edge lengths by $10^{-5}$, strictly restoring positive face area. **Both CPU and GPU pass cleanly in robust mode.**

---

### Category 5: Disconnected Components

#### Case 5.1: Disjoint Closed Surfaces
- **Input Mesh**: [`tests/robustness/meshes/disconnected_components.obj`](file:///DATA/suraj/m1/geometry/geometry-central/tests/robustness/meshes/disconnected_components.obj) (Two disjoint tetrahedra).
- **Reproduction**:
  ```cpp
  auto [mesh, geom] = readManifoldSurfaceMesh("tests/robustness/meshes/disconnected_components.obj");
  HeatMethodDistanceSolver gpuSolver(*geom, 1.0, false, HeatSolverBackend::CUDA_PCG);
  ```
- **Failure Stage**: None (Passes all 5 stages on both CPU and GPU).
- **CPU Behavior**: Computes distance field. On disconnected components, diffused heat $u \equiv 0$, gradient is 0, and distance evaluates to zero.
- **GPU Behavior**: Identical to CPU.
- **Underlying Linear System**:
  - Shifted Laplacian $L + 10^{-6} I$ is block-diagonal with each component having positive diagonal entries.
  - PCG converges robustly across all disconnected components.

---

### Category 6: Very Large Meshes ($> 180\text{k}$ Vertices)

#### Case 6.1: Subdivided Spot ($N = 187,394$ vertices, $NNZ = 1,311,746$)
- **Reproduction**:
  ```cpp
  // Loop-subdivide spot.ply 3 times
  CUDAPCGPositiveDefiniteSolver<double> gpuSolver(heatOp);
  ```
- **Failure Stage**: None.
- **Performance Comparison**:
  - **GPU PCG Execution**: Setup = $0.36$ ms, Solve = $163.70$ ms, Total = **$353.97$ ms** (698 iters, residual $9.98 \times 10^{-7}$).
  - **CPU Direct Cholesky**: Total = **$23,970.50$ ms** ($23.97$ seconds).
  - **Speedup**: **$67.7\times$ faster on GPU**.
- **Memory Footprint**:
  - GPU PCG memory scales strictly as $\mathcal{O}(NNZ + N)$: uses $< 180$ MB VRAM (fits easily in NVIDIA L4 24 GB VRAM).
  - CPU Cholesky scales with fill-in $\mathcal{O}(N^{1.5})$, consuming several gigabytes of RAM.

---

### Category 7: Poorly Conditioned Systems & Regularization

#### Case 7.1: Regularization Shift Parameter Study ($L + \epsilon I$)
Cotan Laplacian condition number as a function of diagonal shift $\epsilon$ on closed mesh:

| Shift ($\epsilon$) | CPU Direct (ms) | GPU PCG (ms) | GPU PCG Iterations | Final Relative Residual | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| $\epsilon = 10^{-2}$ | $3.27$ ms | $9.95$ ms | 70 | $1.00 \times 10^{-6}$ | **Converged** |
| $\epsilon = 10^{-6}$ | $3.24$ ms | $8.47$ ms | 60 | $1.00 \times 10^{-6}$ | **Converged** |
| $\epsilon = 10^{-10}$ | $3.23$ ms | $8.39$ ms | 60 | $1.00 \times 10^{-6}$ | **Converged** |
| $\epsilon = 10^{-14}$ | $3.23$ ms | $8.43$ ms | 60 | $1.00 \times 10^{-6}$ | **Converged** |
| $\epsilon = 0.0$ (pure $L$) | $3.22$ ms | $8.45$ ms | 60 | $1.00 \times 10^{-6}$ | **Converged** (zero-mean RHS) |

- **Observations**:
  - For zero-mean right-hand sides ($\sum b_i = 0$, orthogonal to the 1D null space of constant functions), GPU PCG converges robustly in **60 iterations** even with $\epsilon = 0$.
  - For general non-zero mean RHS, the shift $\epsilon = 10^{-6}$ is necessary to avoid numerical drift along the null space.

---

## 3. Robustness Architecture Summary

| Defect Category | Example Asset | Detection Stage | Default Behavior | Robust Mode Resolution |
| :--- | :--- | :--- | :--- | :--- |
| **Non-Manifold Edge** | `nonmanifold_edge.obj` | Stage 1 (Load) | Throws `duplicate edge` in `ManifoldSurfaceMesh` | `readSurfaceMesh` + `buildIntrinsicTuftedCover` |
| **Non-Manifold Vertex** | `nonmanifold_pinch.obj` | Stage 1 (Load) | Throws `boundary loop` in `ManifoldSurfaceMesh` | `readSurfaceMesh` + `buildIntrinsicTuftedCover` |
| **Self-Edge in Face** | `degenerate_self_edge.obj` | Stage 1 (Load) | Throws `self-edge in face list` | Ingestion filter / `mollifyIntrinsic` |
| **Zero-Area Face** | `zero_area_collinear.obj` | Stage 3 (Matrix) | Throws `Non-finite matrix entry` | `mollifyIntrinsic(1e-5)` + `flipToDelaunay` |
| **Disconnected Mesh** | `disconnected.obj` | Stage 4 (Solve) | Passes without error | Solves each component independently |
| **Very Large Mesh** | `spot_subdiv3x` | Stage 4 (Solve) | CPU Cholesky slow ($>24$ s) | **GPU PCG scales with $67.7\times$ speedup** |
| **Near-Singular System** | Cotan Laplacian | Stage 4 (Solve) | Singular nullspace | Regularization shift $\epsilon = 10^{-6} I$ or zero-mean RHS |

