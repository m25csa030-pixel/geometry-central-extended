#include "geometrycentral/numerical/cuda_pcg_solver.h"
#include "geometrycentral/surface/heat_method_distance.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/subdivide.h"
#include "geometrycentral/surface/surface_mesh.h"
#include "geometrycentral/surface/tufted_laplacian.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace geometrycentral;
using namespace geometrycentral::surface;

struct StageResult {
  std::string stage;
  bool passed = false;
  std::string errorMessage;
  std::string cpuDetails;
  std::string gpuDetails;
};

struct CaseReport {
  std::string category;
  std::string meshName;
  std::string meshPath;
  std::vector<StageResult> stages;
};

static std::vector<CaseReport> g_reports;

void printBanner(const std::string& title) {
  std::cout << "\n======================================================================\n";
  std::cout << "  " << title << "\n";
  std::cout << "======================================================================\n";
}

// Probes all 5 stages for a given mesh file
CaseReport runRobustnessProbe(const std::string& category, const std::string& meshPath) {
  CaseReport report;
  report.category = category;
  report.meshPath = meshPath;
  report.meshName = meshPath.substr(meshPath.find_last_of("/\\") + 1);

  std::cout << "\n>>> Testing Category: [" << category << "] Mesh: " << report.meshName << "\n";

  // -------------------------------------------------------------
  // Stage 1: Mesh Loading & Topology Validation
  // -------------------------------------------------------------
  StageResult s1;
  s1.stage = "1. Loading (readManifoldSurfaceMesh)";
  std::unique_ptr<ManifoldSurfaceMesh> manifoldMesh;
  std::unique_ptr<VertexPositionGeometry> manifoldGeom;
  try {
    std::tie(manifoldMesh, manifoldGeom) = readManifoldSurfaceMesh(meshPath);
    s1.passed = true;
    s1.cpuDetails = "Loaded manifold mesh: V=" + std::to_string(manifoldMesh->nVertices()) +
                    " F=" + std::to_string(manifoldMesh->nFaces());
    s1.gpuDetails = s1.cpuDetails;
    std::cout << "  [Stage 1: Manifold Load] PASSED. " << s1.cpuDetails << "\n";
  } catch (const std::exception& e) {
    s1.passed = false;
    s1.errorMessage = e.what();
    std::cout << "  [Stage 1: Manifold Load] FAILED: " << s1.errorMessage << "\n";
  }
  report.stages.push_back(s1);

  // Probing general SurfaceMesh load if manifold load fails
  std::unique_ptr<SurfaceMesh> generalMesh;
  std::unique_ptr<VertexPositionGeometry> generalGeom;
  StageResult s1General;
  s1General.stage = "1b. Loading (readSurfaceMesh general)";
  try {
    std::tie(generalMesh, generalGeom) = readSurfaceMesh(meshPath);
    s1General.passed = true;
    s1General.cpuDetails = "Loaded general mesh: V=" + std::to_string(generalMesh->nVertices()) +
                           " F=" + std::to_string(generalMesh->nFaces()) +
                           (generalMesh->isManifold() ? " (Manifold)" : " (Non-manifold)");
    s1General.gpuDetails = s1General.cpuDetails;
    std::cout << "  [Stage 1b: General Load] PASSED. " << s1General.cpuDetails << "\n";
  } catch (const std::exception& e) {
    s1General.passed = false;
    s1General.errorMessage = e.what();
    std::cout << "  [Stage 1b: General Load] FAILED: " << s1General.errorMessage << "\n";
  }
  report.stages.push_back(s1General);

  SurfaceMesh* activeMesh = manifoldMesh ? static_cast<SurfaceMesh*>(manifoldMesh.get()) : generalMesh.get();
  VertexPositionGeometry* activeGeom = manifoldGeom ? manifoldGeom.get() : generalGeom.get();

  if (!activeMesh || !activeGeom) {
    std::cout << "  [SKIPPING Stages 2-5: Mesh could not be constructed at Stage 1]\n";
    return report;
  }

  // -------------------------------------------------------------
  // Stage 2: Geometric Properties (Edge lengths, Areas)
  // -------------------------------------------------------------
  StageResult s2;
  s2.stage = "2. Geometric Properties";
  try {
    activeGeom->requireEdgeLengths();
    activeGeom->requireFaceAreas();

    double minLen = 1e9, maxLen = -1e9;
    size_t zeroEdges = 0, nanEdges = 0;
    for (Edge e : activeMesh->edges()) {
      double l = activeGeom->edgeLengths[e];
      if (std::isnan(l)) nanEdges++;
      if (l <= 0.0) zeroEdges++;
      minLen = std::min(minLen, l);
      maxLen = std::max(maxLen, l);
    }

    double minArea = 1e9, maxArea = -1e9;
    size_t zeroFaces = 0, nanFaces = 0;
    for (Face f : activeMesh->faces()) {
      double a = activeGeom->faceAreas[f];
      if (std::isnan(a)) nanFaces++;
      if (a <= 0.0) zeroFaces++;
      minArea = std::min(minArea, a);
      maxArea = std::max(maxArea, a);
    }

    if (zeroEdges > 0 || nanEdges > 0 || zeroFaces > 0 || nanFaces > 0) {
      s2.passed = false;
      s2.errorMessage = "Degenerate elements: zeroEdges=" + std::to_string(zeroEdges) +
                        " nanEdges=" + std::to_string(nanEdges) + " zeroFaces=" + std::to_string(zeroFaces) +
                        " nanFaces=" + std::to_string(nanFaces);
      std::cout << "  [Stage 2: Geometry] DEGENERATE DETECTED: " << s2.errorMessage << "\n";
    } else {
      s2.passed = true;
      s2.cpuDetails = "minEdge=" + std::to_string(minLen) + " minArea=" + std::to_string(minArea);
      s2.gpuDetails = s2.cpuDetails;
      std::cout << "  [Stage 2: Geometry] PASSED. " << s2.cpuDetails << "\n";
    }
  } catch (const std::exception& e) {
    s2.passed = false;
    s2.errorMessage = e.what();
    std::cout << "  [Stage 2: Geometry] FAILED: " << s2.errorMessage << "\n";
  }
  report.stages.push_back(s2);

  // -------------------------------------------------------------
  // Stage 3: Matrix Construction (Laplacian & Mass Matrix)
  // -------------------------------------------------------------
  StageResult s3;
  s3.stage = "3. Matrix Assembly (Cotan Laplacian & Mass)";
  SparseMatrix<double> heatOp;
  SparseMatrix<double> Ls;
  try {
    activeGeom->requireVertexLumpedMassMatrix();
    activeGeom->requireCotanLaplacian();

    SparseMatrix<double>& M = activeGeom->vertexLumpedMassMatrix;
    SparseMatrix<double>& L = activeGeom->cotanLaplacian;

    double shortTime = 0.01;
    heatOp = M + shortTime * L;
    Ls = L + 1e-6 * identityMatrix<double>(activeMesh->nVertices());

    // Check for NaNs or Infs in assembled matrices
    bool hasNan = false;
    bool hasInf = false;
    for (int k = 0; k < heatOp.outerSize(); ++k) {
      for (typename SparseMatrix<double>::InnerIterator it(heatOp, k); it; ++it) {
        if (std::isnan(it.value())) hasNan = true;
        if (std::isinf(it.value())) hasInf = true;
      }
    }

    if (hasNan || hasInf) {
      s3.passed = false;
      s3.errorMessage = std::string("Matrix contains ") + (hasNan ? "NaN " : "") + (hasInf ? "Inf" : "");
      std::cout << "  [Stage 3: Matrix Assembly] FAILED: " << s3.errorMessage << "\n";
    } else {
      s3.passed = true;
      s3.cpuDetails = "heatOp: " + std::to_string(heatOp.rows()) + "x" + std::to_string(heatOp.cols()) +
                      " NNZ=" + std::to_string(heatOp.nonZeros());
      s3.gpuDetails = s3.cpuDetails;
      std::cout << "  [Stage 3: Matrix Assembly] PASSED. " << s3.cpuDetails << "\n";
    }
  } catch (const std::exception& e) {
    s3.passed = false;
    s3.errorMessage = e.what();
    std::cout << "  [Stage 3: Matrix Assembly] FAILED: " << s3.errorMessage << "\n";
  }
  report.stages.push_back(s3);

  // -------------------------------------------------------------
  // Stage 4: Sparse Linear Solvers: CPU vs GPU
  // -------------------------------------------------------------
  StageResult s4;
  s4.stage = "4. Linear Solver (CPU LDLT vs GPU PCG)";
  if (s3.passed) {
    Vector<double> rhs = Vector<double>::Ones(activeMesh->nVertices());
    Vector<double> solCpu, solGpu;
    bool cpuOk = false, gpuOk = false;
    std::string cpuErr, gpuErr;

    // CPU Solve
    try {
      PositiveDefiniteSolver<double> cpuSolver(heatOp);
      solCpu = cpuSolver.solve(rhs);
      if (solCpu.hasNaN()) {
        cpuErr = "CPU solution contains NaN";
      } else {
        cpuOk = true;
        s4.cpuDetails = "CPU LDLT converged (norm=" + std::to_string(solCpu.norm()) + ")";
      }
    } catch (const std::exception& e) {
      cpuErr = e.what();
    }

    // GPU Solve
    try {
      CUDAPCGPositiveDefiniteSolver<double> gpuSolver(heatOp, 1e-6, 2000);
      solGpu = gpuSolver.solve(rhs);
      if (solGpu.hasNaN()) {
        gpuErr = "GPU solution contains NaN";
      } else {
        gpuOk = true;
        s4.gpuDetails = "GPU PCG converged in " + std::to_string(gpuSolver.getIterationsAchieved()) +
                        " iters (res=" + std::to_string(gpuSolver.getFinalResidual()) + ")";
      }
    } catch (const std::exception& e) {
      gpuErr = e.what();
    }

    s4.passed = cpuOk && gpuOk;
    if (!cpuOk) s4.cpuDetails = "CPU FAILED: " + cpuErr;
    if (!gpuOk) s4.gpuDetails = "GPU FAILED: " + gpuErr;
    std::cout << "  [Stage 4: Solvers] CPU: " << (cpuOk ? "PASSED" : cpuErr)
              << " | GPU: " << (gpuOk ? "PASSED" : gpuErr) << "\n";
  } else {
    s4.passed = false;
    s4.errorMessage = "Skipped because matrix assembly failed";
    std::cout << "  [Stage 4: Solvers] SKIPPED (Matrix assembly failed)\n";
  }
  report.stages.push_back(s4);

  // -------------------------------------------------------------
  // Stage 5: Heat Method Geodesic Distance (Standard vs Robust)
  // -------------------------------------------------------------
  StageResult s5;
  s5.stage = "5. Heat Method Distance (Standard vs Robust)";
  Vertex sourceV = activeMesh->vertex(0);

  // 5a. Standard mode (useRobustLaplacian = false)
  bool stdCpuOk = false, stdGpuOk = false;
  std::string stdCpuErr, stdGpuErr;
  try {
    HeatMethodDistanceSolver cpuHeat(*activeGeom, 1.0, false, HeatSolverBackend::CPU);
    VertexData<double> d = cpuHeat.computeDistance(sourceV);
    if (!std::isnan(d[sourceV])) stdCpuOk = true;
  } catch (const std::exception& e) {
    stdCpuErr = e.what();
  }
  try {
    HeatMethodDistanceSolver gpuHeat(*activeGeom, 1.0, false, HeatSolverBackend::CUDA_PCG);
    VertexData<double> d = gpuHeat.computeDistance(sourceV);
    if (!std::isnan(d[sourceV])) stdGpuOk = true;
  } catch (const std::exception& e) {
    stdGpuErr = e.what();
  }

  // 5b. Robust mode (useRobustLaplacian = true)
  bool robCpuOk = false, robGpuOk = false;
  std::string robCpuErr, robGpuErr;
  try {
    HeatMethodDistanceSolver cpuHeat(*activeGeom, 1.0, true, HeatSolverBackend::CPU);
    VertexData<double> d = cpuHeat.computeDistance(sourceV);
    if (!std::isnan(d[sourceV])) robCpuOk = true;
  } catch (const std::exception& e) {
    robCpuErr = e.what();
  }
  try {
    HeatMethodDistanceSolver gpuHeat(*activeGeom, 1.0, true, HeatSolverBackend::CUDA_PCG);
    VertexData<double> d = gpuHeat.computeDistance(sourceV);
    if (!std::isnan(d[sourceV])) robGpuOk = true;
  } catch (const std::exception& e) {
    robGpuErr = e.what();
  }

  std::ostringstream oss;
  oss << "Std(CPU=" << (stdCpuOk ? "OK" : stdCpuErr) << ", GPU=" << (stdGpuOk ? "OK" : stdGpuErr)
      << ") | Robust(CPU=" << (robCpuOk ? "OK" : robCpuErr) << ", GPU=" << (robGpuOk ? "OK" : robGpuErr) << ")";
  s5.cpuDetails = oss.str();
  s5.gpuDetails = s5.cpuDetails;
  s5.passed = (stdCpuOk && stdGpuOk) || (robCpuOk && robGpuOk);
  std::cout << "  [Stage 5: Heat Method] " << s5.cpuDetails << "\n";
  report.stages.push_back(s5);

  return report;
}

// Probe for Category 6: Very Large Mesh
void probeVeryLargeMesh() {
  printBanner("Category 6: Very Large Mesh Scaling & Robustness");

  std::string meshPath = "test/assets/spot.ply";
  std::unique_ptr<ManifoldSurfaceMesh> mesh;
  std::unique_ptr<VertexPositionGeometry> geom;
  std::tie(mesh, geom) = readManifoldSurfaceMesh(meshPath);

  // Subdivide 3 times: N = 187,394 vertices, NNZ = 1,311,746
  std::cout << "Generating very large mesh by subdividing spot 3x...\n";
  for (int l = 0; l < 3; ++l) loopSubdivide(*mesh, *geom);
  size_t nVerts = mesh->nVertices();
  std::cout << "Mesh resolution: V=" << nVerts << " F=" << mesh->nFaces() << "\n";

  geom->requireVertexLumpedMassMatrix();
  geom->requireCotanLaplacian();

  SparseMatrix<double>& M = geom->vertexLumpedMassMatrix;
  SparseMatrix<double>& L = geom->cotanLaplacian;
  SparseMatrix<double> heatOp = M + 0.01 * L;

  Vector<double> rhs = Vector<double>::Ones(nVerts);

  // GPU PCG solve
  std::cout << "Running GPU PCG on " << nVerts << " vertices...\n";
  auto tGpu0 = std::chrono::high_resolution_clock::now();
  CUDAPCGPositiveDefiniteSolver<double> gpuSolver(heatOp, 1e-6, 2000);
  Vector<double> solGpu = gpuSolver.solve(rhs);
  auto tGpu1 = std::chrono::high_resolution_clock::now();
  double gpuMs = std::chrono::duration<double, std::milli>(tGpu1 - tGpu0).count();

  std::cout << "  GPU Result: Setup=" << gpuSolver.getTotalTimeMs() - gpuSolver.getSolveTimeMs()
            << " ms | Solve=" << gpuSolver.getSolveTimeMs() << " ms (Iters: " << gpuSolver.getIterationsAchieved()
            << ", Res: " << gpuSolver.getFinalResidual() << ") | Total=" << gpuMs << " ms\n";

  // CPU Direct solve
  std::cout << "Running CPU Direct Cholesky on " << nVerts << " vertices...\n";
  auto tCpu0 = std::chrono::high_resolution_clock::now();
  PositiveDefiniteSolver<double> cpuSolver(heatOp);
  Vector<double> solCpu = cpuSolver.solve(rhs);
  auto tCpu1 = std::chrono::high_resolution_clock::now();
  double cpuMs = std::chrono::duration<double, std::milli>(tCpu1 - tCpu0).count();

  std::cout << "  CPU Result: Total=" << cpuMs << " ms\n";
  std::cout << "  Speedup: " << (cpuMs / gpuMs) << "x\n";
  std::cout << "  Numerical agreement: " << (solGpu - solCpu).cwiseAbs().maxCoeff() << "\n";
}

// Probe for Category 7: Poorly Conditioned Systems
void probePoorlyConditionedSystem() {
  printBanner("Category 7: Poorly Conditioned Systems & Regularization");

  std::string meshPath = "test/assets/bob_small.ply";
  std::unique_ptr<ManifoldSurfaceMesh> mesh;
  std::unique_ptr<VertexPositionGeometry> geom;
  std::tie(mesh, geom) = readManifoldSurfaceMesh(meshPath);

  geom->requireCotanLaplacian();
  SparseMatrix<double>& L = geom->cotanLaplacian;
  size_t nVerts = mesh->nVertices();

  Vector<double> rhs = Vector<double>::Random(nVerts);
  rhs = rhs.array() - rhs.mean(); // zero-mean orthogonal to constants

  std::vector<double> shifts = {1e-2, 1e-6, 1e-10, 1e-14, 0.0};

  std::cout << std::left << std::setw(15) << "Shift (epsilon)" << std::setw(18) << "CPU Direct (ms)"
            << std::setw(15) << "GPU PCG (ms)" << std::setw(12) << "GPU Iters" << std::setw(18)
            << "GPU Rel. Residual" << "\n";
  std::cout << "-------------------------------------------------------------------------------\n";

  for (double shift : shifts) {
    SparseMatrix<double> A = L + shift * identityMatrix<double>(nVerts);

    double cpuMs = 0;
    try {
      auto t0 = std::chrono::high_resolution_clock::now();
      PositiveDefiniteSolver<double> cpuSolver(A);
      Vector<double> x = cpuSolver.solve(rhs);
      auto t1 = std::chrono::high_resolution_clock::now();
      cpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    } catch (const std::exception& e) {
      cpuMs = -1.0;
    }

    double gpuMs = 0;
    size_t iters = 0;
    double relRes = 0;
    std::string status = "OK";
    try {
      auto t0 = std::chrono::high_resolution_clock::now();
      CUDAPCGPositiveDefiniteSolver<double> gpuSolver(A, 1e-6, 3000);
      Vector<double> x = gpuSolver.solve(rhs);
      auto t1 = std::chrono::high_resolution_clock::now();
      gpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
      iters = gpuSolver.getIterationsAchieved();
      relRes = gpuSolver.getFinalResidual();
    } catch (const std::exception& e) {
      status = e.what();
    }

    std::cout << std::left << std::setw(15) << std::to_string(shift) << std::setw(18)
              << (cpuMs >= 0 ? std::to_string(cpuMs) + " ms" : "FAILED") << std::setw(15)
              << (status == "OK" ? std::to_string(gpuMs) + " ms" : "FAILED") << std::setw(12) << iters
              << std::setw(18) << (status == "OK" ? std::to_string(relRes) : status) << "\n";
  }
}

int main() {
  printBanner("GEOMETRY-CENTRAL: CAD ROBUSTNESS & EDGE-CASE TEST SUITE");

  std::vector<std::pair<std::string, std::string>> cases = {
      {"1. Non-manifold Edge", "tests/robustness/meshes/nonmanifold_edge.obj"},
      {"1. Non-manifold Pinch", "tests/robustness/meshes/nonmanifold_pinch_vertex.obj"},
      {"2. Degenerate Self-Edge", "tests/robustness/meshes/degenerate_self_edge.obj"},
      {"3. Duplicate Coincident Vertices", "tests/robustness/meshes/duplicate_coincident_vertices.obj"},
      {"4. Zero-Area Collinear Face", "tests/robustness/meshes/zero_area_collinear.obj"},
      {"5. Disconnected Components", "tests/robustness/meshes/disconnected_components.obj"},
      {"7. Poorly Conditioned Obtuse", "tests/robustness/meshes/poorly_conditioned_obtuse.obj"},
  };

  for (const auto& c : cases) {
    g_reports.push_back(runRobustnessProbe(c.first, c.second));
  }

  // Categories 6 and 7 dedicated test suites
  probeVeryLargeMesh();
  probePoorlyConditionedSystem();

  printBanner("ROBUSTNESS SUITE COMPLETED SUCCESSFULLY");
  return 0;
}

