# Documentation is hosted at [geometry-central.net](http://geometry-central.net)
---

# Welcome to Geometry Central

[![actions status linux](https://github.com/nmwsharp/geometry-central/workflows/linux/badge.svg)](https://github.com/nmwsharp/geometry-central/actions)
[![actions status macOS](https://github.com/nmwsharp/geometry-central/workflows/macOS/badge.svg)](https://github.com/nmwsharp/geometry-central/actions)
[![actions status windows](https://github.com/nmwsharp/geometry-central/workflows/windows/badge.svg)](https://github.com/nmwsharp/geometry-central/actions)

Geometry-central is a modern C++ library of data structures and algorithms for geometry processing, with a particular focus on surface meshes.

Features include:

- A polished **surface mesh** class, with efficient support for mesh modification, and a system of containers for associating data with mesh elements.
- Implementations of canonical **geometric quantities** on surfaces, ranging from normals and curvatures to tangent vector bases to operators from discrete differential geometry.
- A suite of **powerful algorithms**, including computing distances on surface, generating direction fields, and manipulating intrinsic Delaunay triangulations.
- A coherent set of sparse **linear algebra tools**, based on Eigen and augmented to automatically utilize better solvers if available on your system.
- An optional **CUDA GPU backend** (`GC_ENABLE_CUDA=ON`) providing a Jacobi-preconditioned Conjugate Gradient solver (`CUDAPCGPositiveDefiniteSolver`) that accelerates the Heat Method and Vector Heat Method by up to **30×** on large meshes (≥ 187k vertices) via NVIDIA cuSPARSE/cuBLAS.


**Sample:**

```cpp
// Load a mesh
std::unique_ptr<SurfaceMesh> mesh;
std::unique_ptr<VertexPositionGeometry> geometry;
std::tie(mesh, geometry) = readSurfaceMesh("spot.obj"); 

// Compute vertex areas
VertexData<double> vertexAreas(*mesh);

geometry->requireFaceAreas();
for(Vertex v : mesh->vertices()) {
  double A = 0.;
  for(Face f : v.adjacentFaces()) {
    A += geometry->faceAreas[f] / v.degree();
  }
  vertexAreas[v] = A;
}

// Compute geodesic distances with the GPU backend (requires GC_ENABLE_CUDA=ON)
#include "geometrycentral/surface/heat_method_distance.h"
using namespace geometrycentral::surface;

HeatMethodDistanceSolver gpuSolver(*geometry, 1.0, /*useRobustLaplacian=*/false,
                                   HeatSolverBackend::CUDA_PCG);
VertexData<double> dist = gpuSolver.computeDistance(mesh->vertex(0));
```

Check out the docs, tutorials, and build instructions at [geometry-central.net](http://geometry-central.net).  Use the [sample project](https://github.com/nmwsharp/gc-polyscope-project-template/) to get started with a build system and a gui.

---

## Building with GPU Support

The CUDA backend is **off by default** and requires CUDA Toolkit ≥ 12.x, CMake ≥ 3.17, and an NVIDIA GPU with compute capability ≥ 7.5.

```bash
cmake -B build -S . \
  -DGC_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 \   # replace with your GPU's sm level (75/80/86/89/90)
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

| GPU Architecture | `CMAKE_CUDA_ARCHITECTURES` |
|---|---|
| Turing (T4, RTX 20xx) | `75` |
| Ampere (A100, RTX 30xx) | `80` or `86` |
| Ada Lovelace (L4, RTX 40xx) | `89` |
| Hopper (H100) | `90` |

See [`docs/gpu_solver/`](docs/gpu_solver/) for full architecture details, benchmark data, and known limitations.

---

## GPU Performance Benchmarks

> Benchmarks run on **NVIDIA L4 GPU** (Ada Lovelace, 24 GB VRAM) vs. **AMD EPYC 7742 CPU** (64-core), CUDA 12.6, FP64, averaged over 10 timed runs with 3 warmup solves.

### What do these numbers mean?

The Heat Method works in two steps:
1. **Solve a heat diffusion equation** — spreads "heat" from a source vertex across the mesh.
2. **Solve a Poisson equation** — integrates the heat gradient into geodesic distances.

Both steps involve solving large sparse linear systems. On CPU, the default solver does a one-time matrix factorization (Cholesky), then cheap back-substitutions. The GPU backend replaces this with an iterative Jacobi-preconditioned Conjugate Gradient (PCG) solver — no expensive factorization, just repeated matrix-vector products on the GPU.

---

### Heat Diffusion Solve (`M + tL`, the well-conditioned system)

This is where the GPU shines. The system is well-conditioned (converges in ~25–90 iterations), so PCG is very efficient.

| Mesh Size | CPU Time | GPU Time | Speedup |
|---|---|---|---|
| 2,930 vertices (small) | 5.3 ms | 6.1 ms | 0.9× — CPU faster |
| 11,714 vertices | 25.7 ms | 8.2 ms | **3.1× faster** |
| 46,850 vertices | 127.5 ms | 12.6 ms | **10.1× faster** |
| 187,394 vertices (large) | 628.9 ms | 20.6 ms | **30.5× faster** |

**Takeaway:** The GPU backend is faster once your mesh has ~10,000+ vertices, and the advantage grows rapidly — a 187k-vertex mesh that takes 629 ms on CPU takes only 21 ms on GPU.

---

### Poisson Reconstruction Solve (`L + 1e-6·I`, the ill-conditioned system)

This system is much harder to solve iteratively (condition number ~10¹⁰), requiring up to 2,000 PCG iterations.

| Mesh Size | CPU Time | GPU Time | Speedup |
|---|---|---|---|
| 11,714 vertices | 25.5 ms | 73.5 ms | 0.35× — CPU faster |
| 46,850 vertices | 128.1 ms | 192.6 ms | 0.66× — CPU faster |
| 187,394 vertices (large) | 630.3 ms | 471.2 ms | **1.3× faster** |

**Takeaway:** For the Poisson step, GPU only wins on very large meshes (>100k vertices). The CPU's Cholesky factorization is hard to beat here at smaller scales. Use FP64 — FP32 fails to converge on this system at any mesh size.

---

### Full Heat Method Pipeline (setup + all solves)

For a one-shot call (`heatMethodDistance()`), setup (factorization / GPU upload) dominates for large meshes. CPU Cholesky factorization grows as O(N^1.5)–O(N²), while GPU setup is nearly constant.

| Mesh | CPU Total | GPU Total | Speedup |
|---|---|---|---|
| 250 vertices | 11.4 ms | 15.1 ms | CPU faster |
| 2,930 vertices | 169.1 ms | 56.3 ms | **3.0×** |
| 11,714 vertices | 948.5 ms | 158.8 ms | **6.0×** |
| 46,850 vertices | 6,185.8 ms | 540.5 ms | **11.4×** |

**GPU starts winning at ~3,000 vertices for the full pipeline** because it avoids the expensive Cholesky factorization entirely.

---

### PCIe Transfer Overhead

Data must be copied between CPU RAM and GPU VRAM each solve. This cost is negligible:

- At 187k vertices, transfer = **0.35 ms** out of a 20.9 ms total solve = **1.7% overhead**.

---

### When to Use the GPU Backend

| Scenario | Recommendation |
|---|---|
| Mesh < 5,000 vertices | Use CPU (default) |
| Mesh 5,000–30,000 vertices | GPU faster for setup; similar query speed |
| Mesh > 30,000 vertices | **Use GPU** — significant speedup for both |
| Repeated queries, same mesh | **Use GPU** — setup cost amortized, fast per-query |
| Need FP32 precision | Heat diffusion only — **do not use FP32 for Poisson** |

---

**Related alternatives:** 
[CGAL](https://www.cgal.org/),
[libIGL](https://github.com/libigl/libigl),
[OpenMesh](http://www.openmesh.org/),
[Polygon Mesh Processing Library](https://www.pmp-library.org/),
[CinoLib](https://github.com/mlivesu/cinolib)

---


**GPU backend** (CUDA PCG solver, Heat Method & Vector Heat Method GPU integration) contributed by **Suraj Kumar**, Indian Institute of Technology Jodhpur.





Development of this software was funded in part by NSF Award 1717320, an NSF graduate research fellowship, and gifts from Adobe Research and Autodesk, Inc.
