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

See [`docs/gpu_solver/`](docs/gpu_solver/) for architecture details, benchmark data, and known limitations.

---

**Related alternatives:** 
[CGAL](https://www.cgal.org/),
[libIGL](https://github.com/libigl/libigl),
[OpenMesh](http://www.openmesh.org/),
[Polygon Mesh Processing Library](https://www.pmp-library.org/),
[CinoLib](https://github.com/mlivesu/cinolib)

---
**Additional Information:**
GPU Support provided to this library by Suraj Kumar from Indian Institute Of Technology Jodhpur.


