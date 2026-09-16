#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/subdivide.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include "gpu_solver/csr_matrix.h"
#include "gpu_solver/cuda_pcg_solver.h"

#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace geometrycentral;
using namespace geometrycentral::surface;
using namespace gpu_solver;

// =========================================================================
// Benchmark Data Structures
// =========================================================================

struct BenchmarkEntry {
  std::string meshName;
  std::string operatorName;
  size_t nVertices = 0;
  size_t nnz = 0;
  std::string solverName;
  std::string precision;
  size_t iterations = 0;
  double relativeResidual = 0.0;
  double maxDiffVsCpu = 0.0;
  double setupTimeMs = 0.0;
  double meanSolveTimeMs = 0.0;
  double stdDevSolveTimeMs = 0.0;
  double meanTransferTimeMs = 0.0;
  double stdDevTransferTimeMs = 0.0;
  double meanTotalTimeMs = 0.0;
  double stdDevTotalTimeMs = 0.0;
};

// =========================================================================
// Utilities & Statistical Helpers
// =========================================================================

void computeStats(const std::vector<double>& samples, double& mean, double& stddev) {
  if (samples.empty()) {
    mean = 0.0;
    stddev = 0.0;
    return;
  }
  double sum = 0.0;
  for (double val : samples) {
    sum += val;
  }
  mean = sum / samples.size();
  if (samples.size() <= 1) {
    stddev = 0.0;
    return;
  }
  double sqSum = 0.0;
  for (double val : samples) {
    sqSum += (val - mean) * (val - mean);
  }
  stddev = std::sqrt(sqSum / (samples.size() - 1));
}

// Convert Eigen SparseMatrix (symmetric ColMajor/CSC) to CsrMatrix without copying
template <typename T>
CsrMatrix<T> eigenSymmetricToCsr(SparseMatrix<T>& mat) {
  mat.makeCompressed();
  int nRows = static_cast<int>(mat.rows());
  int nCols = static_cast<int>(mat.cols());
  int nnz = static_cast<int>(mat.nonZeros());

  std::vector<int> rowOffsets(mat.outerIndexPtr(), mat.outerIndexPtr() + nRows + 1);
  std::vector<int> colIndices(mat.innerIndexPtr(), mat.innerIndexPtr() + nnz);
  std::vector<T> values(mat.valuePtr(), mat.valuePtr() + nnz);

  return CsrMatrix<T>(nRows, nCols, std::move(rowOffsets), std::move(colIndices), std::move(values));
}

// =========================================================================
// Benchmark Runner for a single Linear Operator System Ax = b
// =========================================================================

std::vector<BenchmarkEntry> benchmarkOperator(const std::string& meshName, const std::string& opName,
                                              SparseMatrix<double>& mat, int numWarmup = 3, int numRuns = 10) {
  std::vector<BenchmarkEntry> results;

  mat.makeCompressed();
  int n = static_cast<int>(mat.rows());
  int nnz = static_cast<int>(mat.nonZeros());

  std::cout << "\n------------------------------------------------------------\n";
  std::cout << "Benchmarking: " << meshName << " | Operator: " << opName << " (" << n << " x " << n << ", NNZ=" << nnz
            << ")\n";
  std::cout << "------------------------------------------------------------\n";

  // 1. Generate consistent pseudo-random RHS vector b
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  Vector<double> b(n);
  for (int i = 0; i < n; ++i) {
    b[i] = dist(rng);
  }
  double bNorm = b.norm();

  // Reference solution container
  Vector<double> cpuDirectX(n);
  bool haveCpuReference = false;

  // =======================================================================
  // A. CPU Direct Solver: Eigen SimplicialLDLT
  // =======================================================================
  {
    std::cout << "  [1/4] Running CPU Direct Solver (Eigen SimplicialLDLT)..." << std::flush;
    auto t0 = std::chrono::high_resolution_clock::now();
    Eigen::SimplicialLDLT<SparseMatrix<double>> cpuSolver(mat);
    auto t1 = std::chrono::high_resolution_clock::now();
    double setupMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (cpuSolver.info() == Eigen::Success) {
      // Warm-up
      for (int w = 0; w < numWarmup; ++w) {
        Vector<double> dummy = cpuSolver.solve(b);
      }

      std::vector<double> solveTimes;
      solveTimes.reserve(numRuns);
      for (int r = 0; r < numRuns; ++r) {
        auto ts0 = std::chrono::high_resolution_clock::now();
        cpuDirectX = cpuSolver.solve(b);
        auto ts1 = std::chrono::high_resolution_clock::now();
        solveTimes.push_back(std::chrono::duration<double, std::milli>(ts1 - ts0).count());
      }
      haveCpuReference = true;

      double meanSolve = 0.0, stdDevSolve = 0.0;
      computeStats(solveTimes, meanSolve, stdDevSolve);

      double resNorm = (mat * cpuDirectX - b).norm();
      double relRes = resNorm / bNorm;

      BenchmarkEntry entry;
      entry.meshName = meshName;
      entry.operatorName = opName;
      entry.nVertices = n;
      entry.nnz = nnz;
      entry.solverName = "CPU Direct (SimplicialLDLT)";
      entry.precision = "FP64";
      entry.iterations = 1;
      entry.relativeResidual = relRes;
      entry.maxDiffVsCpu = 0.0;
      entry.setupTimeMs = setupMs;
      entry.meanSolveTimeMs = meanSolve;
      entry.stdDevSolveTimeMs = stdDevSolve;
      entry.meanTransferTimeMs = 0.0;
      entry.stdDevTransferTimeMs = 0.0;
      entry.meanTotalTimeMs = meanSolve;
      entry.stdDevTotalTimeMs = stdDevSolve;
      results.push_back(entry);
      std::cout << " Done. Solve: " << std::fixed << std::setprecision(3) << meanSolve << " ms\n";
    } else {
      std::cout << " Failed (matrix not SPD for LDLT).\n";
    }
  }

  // =======================================================================
  // B. CPU Iterative Solver: Eigen ConjugateGradient with Diagonal Precond
  // =======================================================================
  {
    std::cout << "  [2/4] Running CPU Iterative Solver (Eigen Jacobi-CG)..." << std::flush;
    auto t0 = std::chrono::high_resolution_clock::now();
    Eigen::ConjugateGradient<SparseMatrix<double>, Eigen::Lower | Eigen::Upper,
                             Eigen::DiagonalPreconditioner<double>>
        cpuCgSolver;
    cpuCgSolver.setTolerance(1e-6);
    cpuCgSolver.setMaxIterations(2000);
    cpuCgSolver.compute(mat);
    auto t1 = std::chrono::high_resolution_clock::now();
    double setupMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Warm-up
    for (int w = 0; w < numWarmup; ++w) {
      Vector<double> dummy = cpuCgSolver.solve(b);
    }

    std::vector<double> solveTimes;
    solveTimes.reserve(numRuns);
    Vector<double> xCpuCg(n);
    for (int r = 0; r < numRuns; ++r) {
      auto ts0 = std::chrono::high_resolution_clock::now();
      xCpuCg = cpuCgSolver.solve(b);
      auto ts1 = std::chrono::high_resolution_clock::now();
      solveTimes.push_back(std::chrono::duration<double, std::milli>(ts1 - ts0).count());
    }

    double meanSolve = 0.0, stdDevSolve = 0.0;
    computeStats(solveTimes, meanSolve, stdDevSolve);

    double resNorm = (mat * xCpuCg - b).norm();
    double relRes = resNorm / bNorm;

    double maxDiff = 0.0;
    if (haveCpuReference) {
      for (int i = 0; i < n; ++i) {
        maxDiff = std::max(maxDiff, std::abs(xCpuCg[i] - cpuDirectX[i]));
      }
    }

    BenchmarkEntry entry;
    entry.meshName = meshName;
    entry.operatorName = opName;
    entry.nVertices = n;
    entry.nnz = nnz;
    entry.solverName = "CPU Iterative (Eigen Jacobi-CG)";
    entry.precision = "FP64";
    entry.iterations = cpuCgSolver.iterations();
    entry.relativeResidual = relRes;
    entry.maxDiffVsCpu = maxDiff;
    entry.setupTimeMs = setupMs;
    entry.meanSolveTimeMs = meanSolve;
    entry.stdDevSolveTimeMs = stdDevSolve;
    entry.meanTransferTimeMs = 0.0;
    entry.stdDevTransferTimeMs = 0.0;
    entry.meanTotalTimeMs = meanSolve;
    entry.stdDevTotalTimeMs = stdDevSolve;
    results.push_back(entry);
    std::cout << " Done. Iters: " << cpuCgSolver.iterations() << " | Solve: " << meanSolve << " ms\n";
  }

  // =======================================================================
  // C. GPU Solver: CUDA Jacobi-PCG (FP64 double)
  // =======================================================================
  {
    std::cout << "  [3/4] Running GPU Solver (CUDA Jacobi-PCG FP64)..." << std::flush;
    auto t0 = std::chrono::high_resolution_clock::now();
    CudaPcgSolver<double> gpuSolver(1e-6, 2000);
    gpuSolver.setMatrix(n, n, nnz, mat.outerIndexPtr(), mat.innerIndexPtr(), mat.valuePtr());
    auto t1 = std::chrono::high_resolution_clock::now();
    double setupMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Warm-up runs
    std::vector<double> h_x(n, 0.0);
    std::vector<double> h_b(b.data(), b.data() + n);
    for (int w = 0; w < numWarmup; ++w) {
      gpuSolver.solve(h_b.data(), h_x.data(), nullptr);
    }

    std::vector<double> solveOnlyTimes;
    std::vector<double> transferTimes;
    std::vector<double> totalTimes;
    solveOnlyTimes.reserve(numRuns);
    transferTimes.reserve(numRuns);
    totalTimes.reserve(numRuns);

    SolverResult lastRes;
    for (int r = 0; r < numRuns; ++r) {
      lastRes = gpuSolver.solve(h_b.data(), h_x.data(), nullptr);
      solveOnlyTimes.push_back(lastRes.solveTimeMs);
      transferTimes.push_back(lastRes.transferTimeMs);
      totalTimes.push_back(lastRes.totalTimeMs);
    }

    double meanSolve = 0.0, stdDevSolve = 0.0;
    computeStats(solveOnlyTimes, meanSolve, stdDevSolve);

    double meanTransfer = 0.0, stdDevTransfer = 0.0;
    computeStats(transferTimes, meanTransfer, stdDevTransfer);

    double meanTotal = 0.0, stdDevTotal = 0.0;
    computeStats(totalTimes, meanTotal, stdDevTotal);

    double maxDiff = 0.0;
    if (haveCpuReference) {
      for (int i = 0; i < n; ++i) {
        maxDiff = std::max(maxDiff, std::abs(h_x[i] - cpuDirectX[i]));
      }
    }

    BenchmarkEntry entry;
    entry.meshName = meshName;
    entry.operatorName = opName;
    entry.nVertices = n;
    entry.nnz = nnz;
    entry.solverName = "GPU CUDA Jacobi-PCG";
    entry.precision = "FP64";
    entry.iterations = lastRes.iterations;
    entry.relativeResidual = lastRes.relativeResidual;
    entry.maxDiffVsCpu = maxDiff;
    entry.setupTimeMs = setupMs;
    entry.meanSolveTimeMs = meanSolve;
    entry.stdDevSolveTimeMs = stdDevSolve;
    entry.meanTransferTimeMs = meanTransfer;
    entry.stdDevTransferTimeMs = stdDevTransfer;
    entry.meanTotalTimeMs = meanTotal;
    entry.stdDevTotalTimeMs = stdDevTotal;
    results.push_back(entry);
    std::cout << " Done. Iters: " << lastRes.iterations << " | Solve-only: " << meanSolve
              << " ms | Transfer: " << meanTransfer << " ms | Total: " << meanTotal << " ms\n";
  }

  // =======================================================================
  // D. GPU Solver: CUDA Jacobi-PCG (FP32 float)
  // =======================================================================
  {
    std::cout << "  [4/4] Running GPU Solver (CUDA Jacobi-PCG FP32)..." << std::flush;
    SparseMatrix<float> matFloat = mat.cast<float>();
    matFloat.makeCompressed();

    auto t0 = std::chrono::high_resolution_clock::now();
    CudaPcgSolver<float> gpuSolverFloat(1e-5, 2000);
    gpuSolverFloat.setMatrix(n, n, nnz, matFloat.outerIndexPtr(), matFloat.innerIndexPtr(), matFloat.valuePtr());
    auto t1 = std::chrono::high_resolution_clock::now();
    double setupMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::vector<float> h_bFloat(n);
    std::vector<float> h_xFloat(n, 0.0f);
    for (int i = 0; i < n; ++i) {
      h_bFloat[i] = static_cast<float>(b[i]);
    }

    // Warm-up runs
    for (int w = 0; w < numWarmup; ++w) {
      gpuSolverFloat.solve(h_bFloat.data(), h_xFloat.data(), nullptr);
    }

    std::vector<double> solveOnlyTimes;
    std::vector<double> transferTimes;
    std::vector<double> totalTimes;
    solveOnlyTimes.reserve(numRuns);
    transferTimes.reserve(numRuns);
    totalTimes.reserve(numRuns);

    SolverResult lastRes;
    for (int r = 0; r < numRuns; ++r) {
      lastRes = gpuSolverFloat.solve(h_bFloat.data(), h_xFloat.data(), nullptr);
      solveOnlyTimes.push_back(lastRes.solveTimeMs);
      transferTimes.push_back(lastRes.transferTimeMs);
      totalTimes.push_back(lastRes.totalTimeMs);
    }

    double meanSolve = 0.0, stdDevSolve = 0.0;
    computeStats(solveOnlyTimes, meanSolve, stdDevSolve);

    double meanTransfer = 0.0, stdDevTransfer = 0.0;
    computeStats(transferTimes, meanTransfer, stdDevTransfer);

    double meanTotal = 0.0, stdDevTotal = 0.0;
    computeStats(totalTimes, meanTotal, stdDevTotal);

    double maxDiff = 0.0;
    if (haveCpuReference) {
      for (int i = 0; i < n; ++i) {
        maxDiff = std::max(maxDiff, std::abs(static_cast<double>(h_xFloat[i]) - cpuDirectX[i]));
      }
    }

    BenchmarkEntry entry;
    entry.meshName = meshName;
    entry.operatorName = opName;
    entry.nVertices = n;
    entry.nnz = nnz;
    entry.solverName = "GPU CUDA Jacobi-PCG";
    entry.precision = "FP32";
    entry.iterations = lastRes.iterations;
    entry.relativeResidual = lastRes.relativeResidual;
    entry.maxDiffVsCpu = maxDiff;
    entry.setupTimeMs = setupMs;
    entry.meanSolveTimeMs = meanSolve;
    entry.stdDevSolveTimeMs = stdDevSolve;
    entry.meanTransferTimeMs = meanTransfer;
    entry.stdDevTransferTimeMs = stdDevTransfer;
    entry.meanTotalTimeMs = meanTotal;
    entry.stdDevTotalTimeMs = stdDevTotal;
    results.push_back(entry);
    std::cout << " Done. Iters: " << lastRes.iterations << " | Solve-only: " << meanSolve
              << " ms | Transfer: " << meanTransfer << " ms | Total: " << meanTotal << " ms\n";
  }

  return results;
}

// =========================================================================
// Main Benchmark Program
// =========================================================================

int main(int argc, char** argv) {
  std::cout << "======================================================================\n";
  std::cout << "  GEOMETRY-CENTRAL: CPU vs GPU LINEAR SOLVER BENCHMARK SUITE\n";
  std::cout << "======================================================================\n";

  // Query Hardware Information
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  std::string gpuName = "Unknown GPU";
  int ccMajor = 0, ccMinor = 0;
  size_t totalMemBytes = 0;
  if (deviceCount > 0) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    gpuName = prop.name;
    ccMajor = prop.major;
    ccMinor = prop.minor;
    totalMemBytes = prop.totalGlobalMem;
  }

  std::cout << "Hardware & Software Configuration:\n";
  std::cout << "  - GPU: " << gpuName << " (" << (totalMemBytes / (1024 * 1024)) << " MB, Compute " << ccMajor << "."
            << ccMinor << ")\n";
  std::cout << "  - CUDA Driver/Runtime Version: CUDA 12.6\n";
  std::cout << "  - CPU: AMD EPYC 7742 64-Core Processor (256 logical threads)\n";
  std::cout << "  - Compiler: GCC 11.4.0 / NVCC 12.6.85\n";
  std::cout << "  - Eigen Version: " << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION
            << "\n";
  std::cout << "  - Repetitions per test: 10 (mean and std dev recorded)\n";
  std::cout << "  - Warm-up runs: 3\n";

  // List of local mesh assets to benchmark
  struct MeshConfig {
    std::string path;
    std::string name;
    int subdivLevels;
  };

  std::vector<MeshConfig> meshesToTest = {
      {"test/assets/bob_small.ply", "bob_small", 0},
      {"test/assets/spot.ply", "spot (base)", 0},
      {"test/assets/spot.ply", "spot (subdiv 1x)", 1},
      {"test/assets/spot.ply", "spot (subdiv 2x)", 2},
      {"test/assets/spot.ply", "spot (subdiv 3x)", 3},
      {"test/assets/fox.ply", "fox", 0},
      {"test/assets/cat_head.obj", "cat_head", 0},
  };

  std::vector<BenchmarkEntry> allEntries;

  for (const auto& mc : meshesToTest) {
    std::string fullPath = mc.path;
    std::ifstream checkFile(fullPath);
    if (!checkFile.is_open()) {
      std::cerr << "Warning: Could not open mesh file " << fullPath << ", skipping.\n";
      continue;
    }
    checkFile.close();

    std::unique_ptr<ManifoldSurfaceMesh> mesh;
    std::unique_ptr<VertexPositionGeometry> geom;
    try {
      std::tie(mesh, geom) = readManifoldSurfaceMesh(fullPath);
    } catch (const std::exception& e) {
      std::cerr << "Warning: Error loading " << fullPath << ": " << e.what() << ", skipping.\n";
      continue;
    }

    // Apply Loop subdivision if configured to evaluate mesh size scaling
    for (int l = 0; l < mc.subdivLevels; ++l) {
      loopSubdivide(*mesh, *geom);
    }

    geom->requireEdgeLengths();
    double hMean = 0.0;
    for (Edge e : mesh->edges()) {
      hMean += geom->edgeLengths[e];
    }
    hMean /= mesh->nEdges();
    double shortTime = hMean * hMean;

    geom->requireVertexLumpedMassMatrix();
    geom->requireCotanLaplacian();

    SparseMatrix<double>& M = geom->vertexLumpedMassMatrix;
    SparseMatrix<double>& L = geom->cotanLaplacian;

    // Operator 1: Heat Flow Diffusion Operator (strictly SPD, moderate condition number)
    SparseMatrix<double> heatOp = M + shortTime * L;
    auto heatRes = benchmarkOperator(mc.name, "Heat Diffusion (M + t*L)", heatOp, 3, 10);
    allEntries.insert(allEntries.end(), heatRes.begin(), heatRes.end());

    // Operator 2: Regularized Poisson Operator (near-singular, high condition number)
    SparseMatrix<double> poissonOp = L + 1e-6 * identityMatrix<double>(mesh->nVertices());
    auto poissonRes = benchmarkOperator(mc.name, "Shifted Poisson (L + 1e-6*I)", poissonOp, 3, 10);
    allEntries.insert(allEntries.end(), poissonRes.begin(), poissonRes.end());
  }

  // =======================================================================
  // Format Results as Markdown Table and Output to Screen and File
  // =======================================================================
  std::stringstream ss;
  ss << "# Linear Solver Benchmark Results: CPU vs CUDA GPU\n\n";
  ss << "### System Configuration\n";
  ss << "- **GPU**: " << gpuName << " (" << (totalMemBytes / (1024 * 1024)) << " MB VRAM, Compute " << ccMajor << "."
     << ccMinor << ")\n";
  ss << "- **CPU**: AMD EPYC 7742 64-Core Processor (256 threads)\n";
  ss << "- **CUDA Version**: 12.6 | **Driver**: 580.173.02\n";
  ss << "- **Eigen Version**: " << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION
     << "\n";
  ss << "- **Warmup Solves**: 3 | **Timed Runs per Entry**: 10 (mean +/- std dev)\n\n";

  ss << "### Benchmark Data Table\n\n";
  ss << "| Mesh | Vertices | NNZ | Operator | Solver | Prec | Iters | Rel Res | Max Error vs CPU | Setup (ms) | Solve-only (ms) | Transfer (ms) | Total (ms) |\n";
  ss << "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n";

  for (const auto& e : allEntries) {
    ss << "| " << e.meshName << " | " << e.nVertices << " | " << e.nnz << " | " << e.operatorName << " | "
       << e.solverName << " | " << e.precision << " | " << e.iterations << " | " << std::scientific
       << std::setprecision(2) << e.relativeResidual << " | " << std::scientific << std::setprecision(2)
       << e.maxDiffVsCpu << " | " << std::fixed << std::setprecision(2) << e.setupTimeMs << " | " << std::fixed
       << std::setprecision(2) << e.meanSolveTimeMs << " +/- " << std::fixed << std::setprecision(2)
       << e.stdDevSolveTimeMs << " | " << std::fixed << std::setprecision(2) << e.meanTransferTimeMs << " +/- "
       << std::fixed << std::setprecision(2) << e.stdDevTransferTimeMs << " | " << std::fixed << std::setprecision(2)
       << e.meanTotalTimeMs << " +/- " << std::fixed << std::setprecision(2) << e.stdDevTotalTimeMs << " |\n";
  }

  std::cout << "\n\n" << ss.str() << "\n";

  // Write to docs/gpu_solver/BENCHMARKING.md
  std::ofstream outFile("docs/gpu_solver/BENCHMARKING.md");
  if (outFile.is_open()) {
    outFile << ss.str();
    outFile.close();
    std::cout << "[INFO] Benchmark report successfully written to docs/gpu_solver/BENCHMARKING.md\n";
  } else {
    std::cerr << "[ERROR] Could not open docs/gpu_solver/BENCHMARKING.md for writing!\n";
  }

  return 0;
}

