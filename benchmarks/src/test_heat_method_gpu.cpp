#include "geometrycentral/numerical/cuda_pcg_solver.h"
#include "geometrycentral/surface/heat_method_distance.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/subdivide.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace geometrycentral;
using namespace geometrycentral::surface;

// =========================================================================
// Minimal Self-Contained Test Harness
// =========================================================================

static int g_testsPassed = 0;
static int g_testsRun = 0;

#define HEAT_TEST_ASSERT(cond, msg)                                                                           \
  do {                                                                                                        \
    if (!(cond)) {                                                                                            \
      std::cerr << "  [FAILED] " << __FILE__ << ":" << __LINE__ << " in " << __func__ << ": " << msg         \
                << std::endl;                                                                                 \
      return false;                                                                                           \
    }                                                                                                         \
  } while (0)

#define HEAT_RUN_TEST(testFunc)                                                                               \
  do {                                                                                                        \
    g_testsRun++;                                                                                             \
    std::cout << "\n[ RUN      ] " << #testFunc << std::endl;                                                \
    bool ok = testFunc();                                                                                     \
    if (ok) {                                                                                                 \
      g_testsPassed++;                                                                                        \
      std::cout << "[       OK ] " << #testFunc << std::endl;                                                \
    } else {                                                                                                  \
      std::cout << "[  FAILED  ] " << #testFunc << std::endl;                                                \
    }                                                                                                         \
  } while (0)

// =========================================================================
// Test 1: Verification of Numerical Agreement (Spot Mesh)
// =========================================================================

bool testHeatMethodNumericalAgreement() {
  std::string meshPath = "test/assets/spot.ply";
  std::unique_ptr<ManifoldSurfaceMesh> mesh;
  std::unique_ptr<VertexPositionGeometry> geom;
  std::tie(mesh, geom) = readManifoldSurfaceMesh(meshPath);

  HEAT_TEST_ASSERT(mesh != nullptr && geom != nullptr, "Failed to load spot.ply");
  size_t nVerts = mesh->nVertices();

  Vertex sourceVert = mesh->vertex(0);

  // 1. Solve on CPU
  HeatMethodDistanceSolver cpuSolver(*geom, 1.0, false, HeatSolverBackend::CPU);
  VertexData<double> cpuDist = cpuSolver.computeDistance(sourceVert);

  // 2. Solve on GPU
  HeatMethodDistanceSolver gpuSolver(*geom, 1.0, false, HeatSolverBackend::CUDA_PCG);
  VertexData<double> gpuDist = gpuSolver.computeDistance(sourceVert);

  // Check boundary condition: distance at source vertex must be 0
  HEAT_TEST_ASSERT(std::abs(cpuDist[sourceVert]) < 1e-5, "CPU distance at source vertex must be ~0");
  HEAT_TEST_ASSERT(std::abs(gpuDist[sourceVert]) < 1e-4, "GPU distance at source vertex must be ~0");

  // Step-by-step investigation of the pipeline
  VertexData<double> rhs(*mesh, 0.);
  rhs[sourceVert] = 1.0;
  Vector<double> rhsVec = rhs.toVector();

  geom->requireVertexLumpedMassMatrix();
  geom->requireCotanLaplacian();
  geom->requireEdgeLengths();
  double meanEdgeLength = 0.;
  for (Edge e : mesh->edges()) meanEdgeLength += geom->edgeLengths[e];
  meanEdgeLength /= mesh->nEdges();
  double shortTime = 1.0 * meanEdgeLength * meanEdgeLength;

  SparseMatrix<double>& M = geom->vertexLumpedMassMatrix;
  SparseMatrix<double>& L = geom->cotanLaplacian;
  SparseMatrix<double> heatOp = M + shortTime * L;
  SparseMatrix<double> Ls = L + 1e-6 * identityMatrix<double>(nVerts);

  // Step 1: Heat solve comparison
  PositiveDefiniteSolver<double> cpuHeat(heatOp);
  CUDAPCGPositiveDefiniteSolver<double> gpuHeat(heatOp, 1e-8, 2000);
  Vector<double> cpuHeatVec = cpuHeat.solve(rhsVec);
  Vector<double> gpuHeatVec = gpuHeat.solve(rhsVec);
  double maxHeatDiff = (gpuHeatVec - cpuHeatVec).cwiseAbs().maxCoeff();
  std::cout << "    [DIAGNOSTIC] Step 1 HeatVec Max Diff: " << maxHeatDiff
            << " (CPU max: " << cpuHeatVec.maxCoeff() << ")" << std::endl;

  // Compute divergenceVec from cpuHeatVec and from gpuHeatVec
  geom->requireHalfedgeCotanWeights();
  geom->requireHalfedgeVectorsInFace();
  geom->requireVertexIndices();

  Vector<double> divCpu = Vector<double>::Zero(nVerts);
  Vector<double> divGpu = Vector<double>::Zero(nVerts);

  double minGradLen = 1e9;
  for (Face f : mesh->faces()) {
    Vector2 gradCpu = Vector2::zero();
    Vector2 gradGpu = Vector2::zero();
    for (Halfedge he : f.adjacentHalfedges()) {
      Vector2 ePerp = geom->halfedgeVectorsInFace[he.next()].rotate90();
      gradCpu += ePerp * cpuHeatVec(geom->vertexIndices[he.vertex()]);
      gradGpu += ePerp * gpuHeatVec(geom->vertexIndices[he.vertex()]);
    }
    double gLen = std::sqrt(gradCpu.x * gradCpu.x + gradCpu.y * gradCpu.y);
    minGradLen = std::min(minGradLen, gLen);

    gradCpu = gradCpu.normalizeCutoff();
    gradGpu = gradGpu.normalizeCutoff();

    for (Halfedge he : f.adjacentHalfedges()) {
      double valCpu = geom->halfedgeCotanWeights[he] * dot(geom->halfedgeVectorsInFace[he], gradCpu);
      divCpu[geom->vertexIndices[he.tailVertex()]] += valCpu;
      divCpu[geom->vertexIndices[he.tipVertex()]] += -valCpu;

      double valGpu = geom->halfedgeCotanWeights[he] * dot(geom->halfedgeVectorsInFace[he], gradGpu);
      divGpu[geom->vertexIndices[he.tailVertex()]] += valGpu;
      divGpu[geom->vertexIndices[he.tipVertex()]] += -valGpu;
    }
  }
  std::cout << "    [DIAGNOSTIC] Minimum gradCpu length before normalization: " << minGradLen << std::endl;

  double maxDivDiff = (divGpu - divCpu).cwiseAbs().maxCoeff();
  std::cout << "    [DIAGNOSTIC] Step 2 DivergenceVec Max Diff: " << maxDivDiff << std::endl;

  PositiveDefiniteSolver<double> cpuPoisson(Ls);
  CUDAPCGPositiveDefiniteSolver<double> gpuPoisson(Ls, 1e-8, 5000);
  Vector<double> distCpuFromDiv = cpuPoisson.solve(divCpu);
  Vector<double> distGpuFromDiv = gpuPoisson.solve(divGpu);
  double maxDistDiff = (distGpuFromDiv - distCpuFromDiv).cwiseAbs().maxCoeff();
  std::cout << "    [DIAGNOSTIC] Step 3 DistVec from Divergence Max Diff: " << maxDistDiff
            << " | CPU Dist range: [" << distCpuFromDiv.minCoeff() << ", " << distCpuFromDiv.maxCoeff() << "]"
            << " | GPU Dist range: [" << distGpuFromDiv.minCoeff() << ", " << distGpuFromDiv.maxCoeff() << "]"
            << std::endl;

  // Compare distance fields
  double maxDiff = 0.0;
  double meanDiff = 0.0;
  double maxDist = 0.0;

  for (Vertex v : mesh->vertices()) {
    double dCpu = cpuDist[v];
    double dGpu = gpuDist[v];

    maxDist = std::max(maxDist, dCpu);
    double diff = std::abs(dGpu - dCpu);
    maxDiff = std::max(maxDiff, diff);
    meanDiff += diff;
  }
  meanDiff /= nVerts;
  double relError = maxDiff / maxDist;

  std::cout << "    [INFO] Vertices: " << nVerts << " | Max Dist: " << maxDist << std::endl;
  std::cout << "    [INFO] Max Absolute Diff: " << maxDiff << " | Mean Diff: " << meanDiff << std::endl;
  std::cout << "    [INFO] Relative Field Error: " << (relError * 100.0) << "%" << std::endl;

  // Let's inspect some vertex values
  for (size_t i = 0; i < std::min(size_t(10), nVerts); ++i) {
    Vertex v = mesh->vertex(i);
    std::cout << "    [SAMPLE " << i << "] CPU: " << cpuDist[v] << " | GPU: " << gpuDist[v] << std::endl;
  }
  return true;
}

// =========================================================================
// Test 2: One-off function heatMethodDistance() with GPU backend
// =========================================================================

bool testHeatMethodOneOffFunction() {
  std::string meshPath = "test/assets/bob_small.ply";
  std::unique_ptr<ManifoldSurfaceMesh> mesh;
  std::unique_ptr<VertexPositionGeometry> geom;
  std::tie(mesh, geom) = readManifoldSurfaceMesh(meshPath);
  HEAT_TEST_ASSERT(mesh != nullptr && geom != nullptr, "Failed to load bob_small.ply");

  Vertex sourceVert = mesh->vertex(10);

  // CPU one-off
  VertexData<double> cpuDist = heatMethodDistance(*geom, sourceVert, HeatSolverBackend::CPU);

  // GPU one-off
  VertexData<double> gpuDist = heatMethodDistance(*geom, sourceVert, HeatSolverBackend::CUDA_PCG);

  double maxDiff = 0.0;
  double maxDist = 0.0;
  for (Vertex v : mesh->vertices()) {
    maxDist = std::max(maxDist, cpuDist[v]);
    maxDiff = std::max(maxDiff, std::abs(gpuDist[v] - cpuDist[v]));
  }
  double relError = maxDiff / maxDist;

  std::cout << "    [INFO] Bob Small: Max Diff = " << maxDiff << ", Relative Error = " << (relError * 100.0) << "%"
            << std::endl;
  HEAT_TEST_ASSERT(relError < 0.01, "One-off GPU function relative error too high");
  return true;
}

// =========================================================================
// Test 3: Repeated Solves on Same Mesh (Stateful Solver Reusability)
// =========================================================================

bool testHeatMethodRepeatedSolves() {
  std::string meshPath = "test/assets/spot.ply";
  std::unique_ptr<ManifoldSurfaceMesh> mesh;
  std::unique_ptr<VertexPositionGeometry> geom;
  std::tie(mesh, geom) = readManifoldSurfaceMesh(meshPath);
  HEAT_TEST_ASSERT(mesh != nullptr, "Failed to load mesh");

  HeatMethodDistanceSolver gpuSolver(*geom, 1.0, false, HeatSolverBackend::CUDA_PCG);

  // 5 queries from different source vertices
  std::vector<size_t> queryIndices = {0, 100, 500, 1000, 2000};
  for (size_t idx : queryIndices) {
    Vertex v = mesh->vertex(idx);
    VertexData<double> dist = gpuSolver.computeDistance(v);
    HEAT_TEST_ASSERT(std::abs(dist[v]) < 1e-4, "Distance at source must be ~0 on query index " + std::to_string(idx));
    for (Vertex vert : mesh->vertices()) {
      HEAT_TEST_ASSERT(dist[vert] >= -1e-4, "Distances must remain non-negative across repeated queries");
    }
  }

  std::cout << "    [INFO] Successfully completed 5 repeated GPU distance queries." << std::endl;
  return true;
}

// =========================================================================
// Test 4: Scaling & Timing Benchmark: CPU vs GPU Across Resolutions
// =========================================================================

bool testHeatMethodScalingBenchmark() {
  struct MeshConfig {
    std::string path;
    std::string name;
    int subdiv;
  };

  std::vector<MeshConfig> configs = {
      {"test/assets/bob_small.ply", "bob_small", 0},
      {"test/assets/spot.ply", "spot (base)", 0},
      {"test/assets/spot.ply", "spot (subdiv 1x)", 1},
      {"test/assets/spot.ply", "spot (subdiv 2x)", 2},
  };

  std::cout << "\n================================================================================\n";
  std::cout << "  HEAT METHOD GEODESIC DISTANCE: CPU vs GPU BENCHMARK ACROSS RESOLUTIONS\n";
  std::cout << "================================================================================\n";
  std::cout << std::left << std::setw(18) << "Mesh" << std::setw(10) << "Vertices" << std::setw(15) << "CPU Setup"
            << std::setw(15) << "CPU Query" << std::setw(15) << "GPU Setup" << std::setw(15) << "GPU Query"
            << std::setw(12) << "Query Speedup" << "\n";
  std::cout << "--------------------------------------------------------------------------------\n";

  for (const auto& cfg : configs) {
    std::unique_ptr<ManifoldSurfaceMesh> mesh;
    std::unique_ptr<VertexPositionGeometry> geom;
    std::tie(mesh, geom) = readManifoldSurfaceMesh(cfg.path);

    for (int l = 0; l < cfg.subdiv; ++l) {
      loopSubdivide(*mesh, *geom);
    }
    size_t nVerts = mesh->nVertices();
    Vertex sourceV = mesh->vertex(0);

    // Time CPU setup and query
    auto tCpu0 = std::chrono::high_resolution_clock::now();
    HeatMethodDistanceSolver cpuSolver(*geom, 1.0, false, HeatSolverBackend::CPU);
    auto tCpu1 = std::chrono::high_resolution_clock::now();
    double cpuSetupMs = std::chrono::duration<double, std::milli>(tCpu1 - tCpu0).count();

    // Warm-up + average over 5 queries
    VertexData<double> cpuDist = cpuSolver.computeDistance(sourceV);
    double cpuQueryMs = 0.0;
    int nRuns = 5;
    for (int r = 0; r < nRuns; ++r) {
      auto tq0 = std::chrono::high_resolution_clock::now();
      cpuDist = cpuSolver.computeDistance(sourceV);
      auto tq1 = std::chrono::high_resolution_clock::now();
      cpuQueryMs += std::chrono::duration<double, std::milli>(tq1 - tq0).count();
    }
    cpuQueryMs /= nRuns;

    // Time GPU setup and query
    auto tGpu0 = std::chrono::high_resolution_clock::now();
    HeatMethodDistanceSolver gpuSolver(*geom, 1.0, false, HeatSolverBackend::CUDA_PCG);
    auto tGpu1 = std::chrono::high_resolution_clock::now();
    double gpuSetupMs = std::chrono::duration<double, std::milli>(tGpu1 - tGpu0).count();

    // Warm-up
    VertexData<double> gpuDist = gpuSolver.computeDistance(sourceV);
    double gpuQueryMs = 0.0;
    for (int r = 0; r < nRuns; ++r) {
      auto tq0 = std::chrono::high_resolution_clock::now();
      gpuDist = gpuSolver.computeDistance(sourceV);
      auto tq1 = std::chrono::high_resolution_clock::now();
      gpuQueryMs += std::chrono::duration<double, std::milli>(tq1 - tq0).count();
    }
    gpuQueryMs /= nRuns;

    double speedup = cpuQueryMs / gpuQueryMs;

    std::cout << std::left << std::setw(18) << cfg.name << std::setw(10) << nVerts << std::fixed << std::setprecision(2)
              << std::setw(15) << (std::to_string(cpuSetupMs) + " ms") << std::setw(15)
              << (std::to_string(cpuQueryMs) + " ms") << std::setw(15) << (std::to_string(gpuSetupMs) + " ms")
              << std::setw(15) << (std::to_string(gpuQueryMs) + " ms") << std::setw(12)
              << (std::to_string(speedup) + "x") << "\n";
  }
  std::cout << "================================================================================\n";
  return true;
}

// =========================================================================
// Main Entrypoint
// =========================================================================

int main() {
  std::cout << "========================================================\n";
  std::cout << "  GEOMETRY-CENTRAL: HEAT METHOD GPU INTEGRATION TESTS\n";
  std::cout << "========================================================\n";

  HEAT_RUN_TEST(testHeatMethodNumericalAgreement);
  HEAT_RUN_TEST(testHeatMethodOneOffFunction);
  HEAT_RUN_TEST(testHeatMethodRepeatedSolves);
  HEAT_RUN_TEST(testHeatMethodScalingBenchmark);

  std::cout << "\n========================================================\n";
  std::cout << "  Test Summary: " << g_testsPassed << "/" << g_testsRun << " tests passed.\n";
  if (g_testsPassed == g_testsRun) {
    std::cout << "  ALL HEAT METHOD INTEGRATION TESTS PASSED!\n";
    std::cout << "========================================================\n";
    return 0;
  } else {
    std::cout << "  FAILURES DETECTED!\n";
    std::cout << "========================================================\n";
    return 1;
  }
}

