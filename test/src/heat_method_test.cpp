#include "geometrycentral/surface/heat_method_distance.h"
#include "geometrycentral/surface/vector_heat_method.h"
#include "load_test_meshes.h"

#include "gtest/gtest.h"

#ifdef GC_HAVE_CUDA
#include "geometrycentral/numerical/cuda_pcg_solver.h"
#endif

#include <cmath>

using namespace geometrycentral;
using namespace geometrycentral::surface;

class HeatMethodSuite : public MeshAssetSuite {};

// ============================================================
// =============== CPU Heat Method Distance Tests
// ============================================================

TEST_F(HeatMethodSuite, CPUSolverBasic) {
  MeshAsset a = getAsset("bob_small.ply", true);
  Vertex v0 = a.manifoldMesh->vertex(0);

  HeatMethodDistanceSolver solver(*a.geometry, 1.0, false, HeatSolverBackend::CPU);
  VertexData<double> dist = solver.computeDistance(v0);

  EXPECT_NEAR(dist[v0], 0.0, 1e-5);
  for (Vertex v : a.manifoldMesh->vertices()) {
    EXPECT_GE(dist[v], -1e-5);
  }
}

// ============================================================
// =============== GPU Heat Method Distance Tests
// ============================================================

#ifdef GC_HAVE_CUDA

TEST_F(HeatMethodSuite, GPUSolverBasic) {
  MeshAsset a = getAsset("bob_small.ply", true);
  Vertex v0 = a.manifoldMesh->vertex(0);

  HeatMethodDistanceSolver solver(*a.geometry, 1.0, false, HeatSolverBackend::CUDA_PCG);
  VertexData<double> dist = solver.computeDistance(v0);

  EXPECT_NEAR(dist[v0], 0.0, 1e-4);
  for (Vertex v : a.manifoldMesh->vertices()) {
    EXPECT_GE(dist[v], -1e-4);
  }
}

TEST_F(HeatMethodSuite, CPUvsGPUAgreement) {
  MeshAsset a = getAsset("bob_small.ply", true);
  Vertex v0 = a.manifoldMesh->vertex(0);

  HeatMethodDistanceSolver cpuSolver(*a.geometry, 1.0, false, HeatSolverBackend::CPU);
  VertexData<double> cpuDist = cpuSolver.computeDistance(v0);

  HeatMethodDistanceSolver gpuSolver(*a.geometry, 1.0, false, HeatSolverBackend::CUDA_PCG);
  VertexData<double> gpuDist = gpuSolver.computeDistance(v0);

  double maxDiff = 0.0;
  double maxVal = 0.0;
  for (Vertex v : a.manifoldMesh->vertices()) {
    maxVal = std::max(maxVal, cpuDist[v]);
    maxDiff = std::max(maxDiff, std::abs(gpuDist[v] - cpuDist[v]));
  }
  double relError = maxDiff / maxVal;
  EXPECT_LT(relError, 0.01); // Relative error < 1%
}

TEST_F(HeatMethodSuite, OneOffFunction) {
  MeshAsset a = getAsset("bob_small.ply", true);
  Vertex v0 = a.manifoldMesh->vertex(5);

  VertexData<double> cpuDist = heatMethodDistance(*a.geometry, v0, HeatSolverBackend::CPU);
  VertexData<double> gpuDist = heatMethodDistance(*a.geometry, v0, HeatSolverBackend::CUDA_PCG);

  double maxDiff = 0.0;
  double maxVal = 0.0;
  for (Vertex v : a.manifoldMesh->vertices()) {
    maxVal = std::max(maxVal, cpuDist[v]);
    maxDiff = std::max(maxDiff, std::abs(gpuDist[v] - cpuDist[v]));
  }
  EXPECT_LT(maxDiff / maxVal, 0.01);
}

TEST_F(HeatMethodSuite, RepeatedSolvesMultipleSources) {
  MeshAsset a = getAsset("bob_small.ply", true);
  HeatMethodDistanceSolver gpuSolver(*a.geometry, 1.0, false, HeatSolverBackend::CUDA_PCG);

  for (size_t i = 0; i < 5; ++i) {
    Vertex v = a.manifoldMesh->vertex(i * 10);
    VertexData<double> dist = gpuSolver.computeDistance(v);
    EXPECT_NEAR(dist[v], 0.0, 1e-4);
  }
}

// ============================================================
// =============== Vector Heat Method GPU Tests
// ============================================================

TEST_F(HeatMethodSuite, VectorHeat_ScalarExtension) {
  MeshAsset a = getAsset("bob_small.ply", true);
  Vertex v0 = a.manifoldMesh->vertex(0);

  VectorHeatMethodSolver cpuSolver(*a.geometry, 1.0, VectorHeatSolverBackend::CPU);
  VectorHeatMethodSolver gpuSolver(*a.geometry, 1.0, VectorHeatSolverBackend::CUDA_PCG);

  std::vector<std::tuple<Vertex, double>> sources = {{v0, 42.0}};
  VertexData<double> cpuExt = cpuSolver.extendScalar(sources);
  VertexData<double> gpuExt = gpuSolver.extendScalar(sources);

  // Values across the surface should be close to 42.0
  EXPECT_NEAR(gpuExt[v0], 42.0, 1e-3);

  double maxDiff = 0.0;
  for (Vertex v : a.manifoldMesh->vertices()) {
    maxDiff = std::max(maxDiff, std::abs(gpuExt[v] - cpuExt[v]));
  }
  EXPECT_LT(maxDiff, 0.1); // Strong agreement
}

TEST_F(HeatMethodSuite, VectorHeat_TangentTransport) {
  MeshAsset a = getAsset("bob_small.ply", true);
  Vertex v0 = a.manifoldMesh->vertex(0);
  Vector2 sourceVec{1.0, 0.5};

  VectorHeatMethodSolver cpuSolver(*a.geometry, 1.0, VectorHeatSolverBackend::CPU);
  VectorHeatMethodSolver gpuSolver(*a.geometry, 1.0, VectorHeatSolverBackend::CUDA_PCG);

  VertexData<Vector2> cpuTransport = cpuSolver.transportTangentVector(v0, sourceVec);
  VertexData<Vector2> gpuTransport = gpuSolver.transportTangentVector(v0, sourceVec);

  double expectedNorm = sourceVec.norm();
  for (Vertex v : a.manifoldMesh->vertices()) {
    EXPECT_NEAR(gpuTransport[v].norm(), expectedNorm, 1e-4);
  }

  double maxDiff = 0.0;
  for (Vertex v : a.manifoldMesh->vertices()) {
    maxDiff = std::max(maxDiff, (gpuTransport[v] - cpuTransport[v]).norm());
  }
  EXPECT_LT(maxDiff, 0.05);
}

TEST_F(HeatMethodSuite, VectorHeat_LogMap) {
  MeshAsset a = getAsset("bob_small.ply", true);
  Vertex v0 = a.manifoldMesh->vertex(0);

  VectorHeatMethodSolver cpuSolver(*a.geometry, 1.0, VectorHeatSolverBackend::CPU);
  VectorHeatMethodSolver gpuSolver(*a.geometry, 1.0, VectorHeatSolverBackend::CUDA_PCG);

  VertexData<Vector2> cpuLog = cpuSolver.computeLogMap(v0, LogMapStrategy::VectorHeat);
  VertexData<Vector2> gpuLog = gpuSolver.computeLogMap(v0, LogMapStrategy::VectorHeat);

  // At the source, log map must be zero
  EXPECT_NEAR(gpuLog[v0].norm(), 0.0, 1e-4);
  EXPECT_NEAR(cpuLog[v0].norm(), 0.0, 1e-4);

  double maxDiff = 0.0;
  double maxNorm = 0.0;
  for (Vertex v : a.manifoldMesh->vertices()) {
    maxNorm = std::max(maxNorm, cpuLog[v].norm());
    maxDiff = std::max(maxDiff, (gpuLog[v] - cpuLog[v]).norm());
  }
  EXPECT_LT(maxDiff / maxNorm, 0.05); // Relative error < 5%
}

#endif


