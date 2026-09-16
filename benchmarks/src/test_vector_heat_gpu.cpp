#include "geometrycentral/numerical/cuda_pcg_solver.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/subdivide.h"
#include "geometrycentral/surface/vector_heat_method.h"
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

#define VHEAT_TEST_ASSERT(cond, msg)                                                                           \
  do {                                                                                                         \
    if (!(cond)) {                                                                                             \
      std::cerr << "  [FAILED] " << __FILE__ << ":" << __LINE__ << " in " << __func__ << ": " << msg          \
                << std::endl;                                                                                  \
      return false;                                                                                            \
    }                                                                                                          \
  } while (0)

#define VHEAT_RUN_TEST(testFunc)                                                                               \
  do {                                                                                                         \
    g_testsRun++;                                                                                              \
    std::cout << "\n[ RUN      ] " << #testFunc << std::endl;                                                 \
    bool ok = testFunc();                                                                                      \
    if (ok) {                                                                                                  \
      g_testsPassed++;                                                                                         \
      std::cout << "[       OK ] " << #testFunc << std::endl;                                                 \
    } else {                                                                                                   \
      std::cout << "[  FAILED  ] " << #testFunc << std::endl;                                                 \
    }                                                                                                          \
  } while (0)

// =========================================================================
// Test 1: Complex Synthetic Hermitian SPD PCG Test
// =========================================================================

bool testComplexSyntheticHermitianPCG() {
  const size_t N = 4;
  SparseMatrix<std::complex<double>> mat(N, N);
  std::vector<Eigen::Triplet<std::complex<double>>> triplets;

  // Diagonal elements (strictly positive real)
  triplets.emplace_back(0, 0, std::complex<double>(5.0, 0.0));
  triplets.emplace_back(1, 1, std::complex<double>(6.0, 0.0));
  triplets.emplace_back(2, 2, std::complex<double>(7.0, 0.0));
  triplets.emplace_back(3, 3, std::complex<double>(8.0, 0.0));

  // Off-diagonal Hermitian pairs (A_ij = conj(A_ji))
  triplets.emplace_back(0, 1, std::complex<double>(1.0, 1.0));
  triplets.emplace_back(1, 0, std::complex<double>(1.0, -1.0));

  triplets.emplace_back(1, 2, std::complex<double>(0.5, -0.5));
  triplets.emplace_back(2, 1, std::complex<double>(0.5, 0.5));

  triplets.emplace_back(2, 3, std::complex<double>(1.0, 0.0));
  triplets.emplace_back(3, 2, std::complex<double>(1.0, 0.0));

  mat.setFromTriplets(triplets.begin(), triplets.end());
  mat.makeCompressed();

  Vector<std::complex<double>> exactX(N);
  exactX << std::complex<double>(1.0, 2.0), std::complex<double>(-1.0, 0.5), std::complex<double>(3.0, -1.5),
      std::complex<double>(0.0, 4.0);

  Vector<std::complex<double>> b = mat * exactX;

  ComplexCUDAPCGPositiveDefiniteSolver solver(mat, 1e-10, 100);
  Vector<std::complex<double>> solX = solver.solve(b);

  double maxDiff = (solX - exactX).cwiseAbs().maxCoeff();
  std::cout << "    [INFO] Complex Synthetic PCG Max Diff: " << maxDiff
            << " (Iters: " << solver.getIterationsAchieved() << ", Res: " << solver.getFinalResidual() << ")"
            << std::endl;

  VHEAT_TEST_ASSERT(maxDiff < 1e-6, "Complex PCG synthetic solution error too high");
  return true;
}

// =========================================================================
// Test 2: Vector Heat Method Numerical Agreement (Bob Small)
// =========================================================================

bool testVectorHeatNumericalAgreement() {
  std::string meshPath = "test/assets/bob_small.ply";
  std::unique_ptr<ManifoldSurfaceMesh> mesh;
  std::unique_ptr<VertexPositionGeometry> geom;
  std::tie(mesh, geom) = readManifoldSurfaceMesh(meshPath);
  VHEAT_TEST_ASSERT(mesh != nullptr && geom != nullptr, "Failed to load bob_small.ply");

  Vertex v0 = mesh->vertex(0);

  VectorHeatMethodSolver cpuSolver(*geom, 1.0, VectorHeatSolverBackend::CPU);
  VectorHeatMethodSolver gpuSolver(*geom, 1.0, VectorHeatSolverBackend::CUDA_PCG);

  // 1. Scalar Extension
  std::vector<std::tuple<Vertex, double>> sources = {{v0, 100.0}};
  VertexData<double> cpuExt = cpuSolver.extendScalar(sources);
  VertexData<double> gpuExt = gpuSolver.extendScalar(sources);

  double maxExtDiff = 0.0;
  for (Vertex v : mesh->vertices()) {
    maxExtDiff = std::max(maxExtDiff, std::abs(gpuExt[v] - cpuExt[v]));
  }
  std::cout << "    [INFO] Scalar Extension Max Diff: " << maxExtDiff << std::endl;
  VHEAT_TEST_ASSERT(maxExtDiff < 0.1, "Scalar extension diff too large");

  // 2. Tangent Vector Transport
  Vector2 srcVec{2.0, 1.0};
  VertexData<Vector2> cpuTrans = cpuSolver.transportTangentVector(v0, srcVec);
  VertexData<Vector2> gpuTrans = gpuSolver.transportTangentVector(v0, srcVec);

  double maxTransDiff = 0.0;
  for (Vertex v : mesh->vertices()) {
    maxTransDiff = std::max(maxTransDiff, (gpuTrans[v] - cpuTrans[v]).norm());
  }
  double relTransDiff = maxTransDiff / srcVec.norm();
  std::cout << "    [INFO] Tangent Transport Max Diff: " << maxTransDiff << ", Relative: " << (relTransDiff * 100.0) << "%"
            << std::endl;
  VHEAT_TEST_ASSERT(relTransDiff < 0.05, "Tangent transport diff too large");

  // 3. Log Map
  VertexData<Vector2> cpuLog = cpuSolver.computeLogMap(v0, LogMapStrategy::VectorHeat);
  VertexData<Vector2> gpuLog = gpuSolver.computeLogMap(v0, LogMapStrategy::VectorHeat);

  double maxLogDiff = 0.0;
  double maxLogNorm = 0.0;
  for (Vertex v : mesh->vertices()) {
    maxLogNorm = std::max(maxLogNorm, cpuLog[v].norm());
    maxLogDiff = std::max(maxLogDiff, (gpuLog[v] - cpuLog[v]).norm());
  }
  double relLogDiff = maxLogDiff / maxLogNorm;
  std::cout << "    [INFO] Log Map Max Diff: " << maxLogDiff << ", Relative: " << (relLogDiff * 100.0) << "%"
            << std::endl;
  VHEAT_TEST_ASSERT(relLogDiff < 0.05, "Log map relative error too large");

  return true;
}

// =========================================================================
// Test 3: Repeated Solves across multiple source queries
// =========================================================================

bool testVectorHeatRepeatedSolves() {
  std::string meshPath = "test/assets/spot.ply";
  std::unique_ptr<ManifoldSurfaceMesh> mesh;
  std::unique_ptr<VertexPositionGeometry> geom;
  std::tie(mesh, geom) = readManifoldSurfaceMesh(meshPath);
  VHEAT_TEST_ASSERT(mesh != nullptr, "Failed to load spot.ply");

  VectorHeatMethodSolver cpuSolver(*geom, 1.0, VectorHeatSolverBackend::CPU);
  VectorHeatMethodSolver gpuSolver(*geom, 1.0, VectorHeatSolverBackend::CUDA_PCG);

  std::vector<size_t> queryIndices = {0, 100, 500, 1000, 2000};
  for (size_t idx : queryIndices) {
    Vertex v = mesh->vertex(idx);
    Vector2 srcVec{1.0, 0.0};
    VertexData<Vector2> cpuTrans = cpuSolver.transportTangentVector(v, srcVec);
    VertexData<Vector2> gpuTrans = gpuSolver.transportTangentVector(v, srcVec);

    double minNormGpu = 1e9, maxNormGpu = -1e9;
    size_t nanCountGpu = 0;
    double minNormCpu = 1e9, maxNormCpu = -1e9;
    size_t nanCountCpu = 0;
    for (Vertex vert : mesh->vertices()) {
      double ng = gpuTrans[vert].norm();
      if (std::isnan(ng)) nanCountGpu++;
      minNormGpu = std::min(minNormGpu, ng);
      maxNormGpu = std::max(maxNormGpu, ng);

      double nc = cpuTrans[vert].norm();
      if (std::isnan(nc)) nanCountCpu++;
      minNormCpu = std::min(minNormCpu, nc);
      maxNormCpu = std::max(maxNormCpu, nc);
    }
    std::cout << "    [DIAGNOSTIC] Query " << idx
              << " | CPU: min=" << minNormCpu << " max=" << maxNormCpu << " NaNs=" << nanCountCpu
              << " | GPU: min=" << minNormGpu << " max=" << maxNormGpu << " NaNs=" << nanCountGpu << std::endl;

    VHEAT_TEST_ASSERT(nanCountGpu == 0, "GPU transport produced NaNs on query " + std::to_string(idx));
    VHEAT_TEST_ASSERT(std::abs(gpuTrans[v].norm() - 1.0) < 1e-4, "Norm at source must be 1.0");
  }

  std::cout << "    [INFO] Successfully executed 5 repeated GPU tangent transport queries." << std::endl;
  return true;
}

// =========================================================================
// Test 4: Comprehensive Scaling & Timing Benchmark: CPU vs GPU
// =========================================================================

bool testVectorHeatScalingBenchmark() {
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

  std::cout << "\n===================================================================================================\n";
  std::cout << "  VECTOR HEAT METHOD: TANGENT VECTOR TRANSPORT BENCHMARK (CPU vs CUDA GPU)\n";
  std::cout << "===================================================================================================\n";
  std::cout << std::left << std::setw(18) << "Mesh" << std::setw(10) << "Vertices" << std::setw(15) << "CPU Setup"
            << std::setw(15) << "CPU Query" << std::setw(15) << "GPU Setup" << std::setw(15) << "GPU Query"
            << std::setw(15) << "Query Speedup" << "\n";
  std::cout << "---------------------------------------------------------------------------------------------------\n";

  for (const auto& cfg : configs) {
    std::unique_ptr<ManifoldSurfaceMesh> mesh;
    std::unique_ptr<VertexPositionGeometry> geom;
    std::tie(mesh, geom) = readManifoldSurfaceMesh(cfg.path);

    for (int l = 0; l < cfg.subdiv; ++l) {
      loopSubdivide(*mesh, *geom);
    }
    size_t nVerts = mesh->nVertices();
    Vertex sourceV = mesh->vertex(0);
    Vector2 sourceVec{1.0, 0.5};

    // Time CPU setup
    auto tCpu0 = std::chrono::high_resolution_clock::now();
    VectorHeatMethodSolver cpuSolver(*geom, 1.0, VectorHeatSolverBackend::CPU);
    // Trigger ensureHaveVectorHeatSolver by calling once
    VertexData<Vector2> cpuRes = cpuSolver.transportTangentVector(sourceV, sourceVec);
    auto tCpu1 = std::chrono::high_resolution_clock::now();
    double cpuSetupAndWarmupMs = std::chrono::duration<double, std::milli>(tCpu1 - tCpu0).count();

    // Average CPU query time over 5 runs
    double cpuQueryMs = 0.0;
    int nRuns = 5;
    for (int r = 0; r < nRuns; ++r) {
      auto tq0 = std::chrono::high_resolution_clock::now();
      cpuRes = cpuSolver.transportTangentVector(sourceV, sourceVec);
      auto tq1 = std::chrono::high_resolution_clock::now();
      cpuQueryMs += std::chrono::duration<double, std::milli>(tq1 - tq0).count();
    }
    cpuQueryMs /= nRuns;

    // Time GPU setup
    auto tGpu0 = std::chrono::high_resolution_clock::now();
    VectorHeatMethodSolver gpuSolver(*geom, 1.0, VectorHeatSolverBackend::CUDA_PCG);
    VertexData<Vector2> gpuRes = gpuSolver.transportTangentVector(sourceV, sourceVec);
    auto tGpu1 = std::chrono::high_resolution_clock::now();
    double gpuSetupAndWarmupMs = std::chrono::duration<double, std::milli>(tGpu1 - tGpu0).count();

    // Average GPU query time over 5 runs
    double gpuQueryMs = 0.0;
    for (int r = 0; r < nRuns; ++r) {
      auto tq0 = std::chrono::high_resolution_clock::now();
      gpuRes = gpuSolver.transportTangentVector(sourceV, sourceVec);
      auto tq1 = std::chrono::high_resolution_clock::now();
      gpuQueryMs += std::chrono::duration<double, std::milli>(tq1 - tq0).count();
    }
    gpuQueryMs /= nRuns;

    double speedup = cpuQueryMs / gpuQueryMs;

    std::cout << std::left << std::setw(18) << cfg.name << std::setw(10) << nVerts << std::fixed
              << std::setprecision(2) << std::setw(15) << (std::to_string(cpuSetupAndWarmupMs) + " ms")
              << std::setw(15) << (std::to_string(cpuQueryMs) + " ms") << std::setw(15)
              << (std::to_string(gpuSetupAndWarmupMs) + " ms") << std::setw(15)
              << (std::to_string(gpuQueryMs) + " ms") << std::setw(15) << (std::to_string(speedup) + "x") << "\n";
  }
  std::cout << "===================================================================================================\n";
  return true;
}

// =========================================================================
// Test 5: Solve-Only vs PCIe Transfer Time Breakdown
// =========================================================================

bool testVectorHeatSolveVsTransfer() {
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

  std::cout << "\n===================================================================================================\n";
  std::cout << "  GPU COMPLEX VECTOR HEAT SOLVER: SOLVE-ONLY vs TRANSFER TIME BREAKDOWN\n";
  std::cout << "===================================================================================================\n";
  std::cout << std::left << std::setw(18) << "Mesh" << std::setw(10) << "Vertices" << std::setw(10) << "Iters"
            << std::setw(18) << "Solve-Only (ms)" << std::setw(18) << "Transfer (ms)" << std::setw(18)
            << "Total GPU (ms)" << std::setw(15) << "Transfer %" << "\n";
  std::cout << "---------------------------------------------------------------------------------------------------\n";

  for (const auto& cfg : configs) {
    std::unique_ptr<ManifoldSurfaceMesh> mesh;
    std::unique_ptr<VertexPositionGeometry> geom;
    std::tie(mesh, geom) = readManifoldSurfaceMesh(cfg.path);

    for (int l = 0; l < cfg.subdiv; ++l) {
      loopSubdivide(*mesh, *geom);
    }
    size_t nVerts = mesh->nVertices();

    geom->requireEdgeLengths();
    geom->requireVertexLumpedMassMatrix();
    geom->requireVertexConnectionLaplacian();

    double meanEdgeLength = 0.;
    for (Edge e : mesh->edges()) meanEdgeLength += geom->edgeLengths[e];
    meanEdgeLength /= mesh->nEdges();
    double shortTime = 1.0 * meanEdgeLength * meanEdgeLength;

    SparseMatrix<std::complex<double>>& Lconn = geom->vertexConnectionLaplacian;
    SparseMatrix<std::complex<double>> vectorOp =
        geom->vertexLumpedMassMatrix.cast<std::complex<double>>() + shortTime * Lconn;

    ComplexCUDAPCGPositiveDefiniteSolver solver(vectorOp, 1e-6, 2000);
    Vector<std::complex<double>> rhs = Vector<std::complex<double>>::Zero(nVerts);
    rhs[0] = std::complex<double>(1.0, 0.0);

    // Warm-up solve
    solver.solve(rhs);

    // Timed solve
    solver.solve(rhs);

    double solveMs = solver.getSolveTimeMs();
    double transferMs = solver.getTransferTimeMs();
    double totalMs = solver.getTotalTimeMs();
    double transferPct = (transferMs / totalMs) * 100.0;
    size_t iters = solver.getIterationsAchieved();

    std::cout << std::left << std::setw(18) << cfg.name << std::setw(10) << nVerts << std::setw(10) << iters
              << std::fixed << std::setprecision(3) << std::setw(18) << solveMs << std::setw(18) << transferMs
              << std::setw(18) << totalMs << std::setprecision(2) << std::setw(15)
              << (std::to_string(transferPct).substr(0, 4) + " %") << "\n";
  }
  std::cout << "===================================================================================================\n";
  return true;
}

// =========================================================================
// Main Entrypoint
// =========================================================================

int main() {
  std::cout << "========================================================\n";
  std::cout << "  GEOMETRY-CENTRAL: VECTOR HEAT METHOD GPU TESTS\n";
  std::cout << "========================================================\n";

  VHEAT_RUN_TEST(testComplexSyntheticHermitianPCG);
  VHEAT_RUN_TEST(testVectorHeatNumericalAgreement);
  VHEAT_RUN_TEST(testVectorHeatRepeatedSolves);
  VHEAT_RUN_TEST(testVectorHeatScalingBenchmark);
  VHEAT_RUN_TEST(testVectorHeatSolveVsTransfer);

  std::cout << "\n========================================================\n";
  std::cout << "  Test Summary: " << g_testsPassed << "/" << g_testsRun << " tests passed.\n";
  if (g_testsPassed == g_testsRun) {
    std::cout << "  ALL VECTOR HEAT METHOD INTEGRATION TESTS PASSED!\n";
    std::cout << "========================================================\n";
    return 0;
  } else {
    std::cout << "  FAILURES DETECTED!\n";
    std::cout << "========================================================\n";
    return 1;
  }
}


