#include "gpu_solver/csr_matrix.h"
#include "gpu_solver/cuda_pcg_solver.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace gpu_solver;

// =========================================================================
// Minimal Self-Contained Test Harness
// =========================================================================

static int g_numTestsRun = 0;
static int g_numTestsPassed = 0;
static int g_numTestsFailed = 0;

#define TEST_ASSERT(cond, msg)                                                                               \
  do {                                                                                                       \
    if (!(cond)) {                                                                                           \
      std::cerr << "  [FAILED] " << __FILE__ << ":" << __LINE__ << " in " << __func__ << ": " << msg        \
                << std::endl;                                                                                \
      return false;                                                                                          \
    }                                                                                                        \
  } while (0)

#define TEST_ASSERT_NEAR(val1, val2, tol, msg)                                                               \
  do {                                                                                                       \
    double diff = std::abs(static_cast<double>(val1) - static_cast<double>(val2));                           \
    if (diff > (tol)) {                                                                                      \
      std::cerr << "  [FAILED] " << __FILE__ << ":" << __LINE__ << " in " << __func__ << ": " << msg        \
                << " (expected: " << (val2) << ", got: " << (val1) << ", diff: " << diff                    \
                << " > tol: " << (tol) << ")" << std::endl;                                                  \
      return false;                                                                                          \
    }                                                                                                        \
  } while (0)

#define RUN_TEST(testFunc)                                                                                   \
  do {                                                                                                       \
    g_numTestsRun++;                                                                                         \
    std::cout << "[ RUN      ] " << #testFunc << std::endl;                                                 \
    bool ok = testFunc();                                                                                    \
    if (ok) {                                                                                                \
      g_numTestsPassed++;                                                                                    \
      std::cout << "[       OK ] " << #testFunc << std::endl;                                                \
    } else {                                                                                                 \
      g_numTestsFailed++;                                                                                    \
      std::cout << "[  FAILED  ] " << #testFunc << std::endl;                                                \
    }                                                                                                        \
  } while (0)

// =========================================================================
// Helper Functions
// =========================================================================

// Compute residual norm on CPU: ||A * x - b||_2
template <typename T>
double computeCpuResidualNorm(const CsrMatrix<T>& A, const std::vector<T>& x, const std::vector<T>& b) {
  std::vector<T> Ax(A.nRows, 0);
  A.spmv(x.data(), Ax.data());
  double sumSq = 0;
  for (int i = 0; i < A.nRows; ++i) {
    double diff = static_cast<double>(Ax[i]) - static_cast<double>(b[i]);
    sumSq += diff * diff;
  }
  return std::sqrt(sumSq);
}

// Convert CsrMatrix to Eigen::SparseMatrix
template <typename T>
Eigen::SparseMatrix<T> csrToEigen(const CsrMatrix<T>& csr) {
  Eigen::SparseMatrix<T> mat(csr.nRows, csr.nCols);
  std::vector<Eigen::Triplet<T>> triplets;
  triplets.reserve(csr.nnz);
  for (int i = 0; i < csr.nRows; ++i) {
    for (int j = csr.rowOffsets[i]; j < csr.rowOffsets[i + 1]; ++j) {
      triplets.emplace_back(i, csr.colIndices[j], csr.values[j]);
    }
  }
  mat.setFromTriplets(triplets.begin(), triplets.end());
  mat.makeCompressed();
  return mat;
}

// =========================================================================
// Tests
// =========================================================================

// 1. Diagonal SPD Matrix (Jacobi should converge in 1 iteration!)
bool testDiagonalSpd() {
  int n = 100;
  std::vector<double> diag(n);
  for (int i = 0; i < n; ++i) {
    diag[i] = 1.0 + 3.0 * (i + 1); // strictly positive diagonal
  }
  CsrMatrix<double> mat = CsrMatrix<double>::fromDiagonal(diag);

  std::vector<double> b(n);
  for (int i = 0; i < n; ++i) {
    b[i] = 2.0 * (i + 1);
  }

  CudaPcgSolver<double> solver(mat, 1e-8, 100);
  std::vector<double> x(n);
  SolverResult res = solver.solve(b, x);

  TEST_ASSERT(res.hasConverged(), "Solver should converge on diagonal matrix.");
  TEST_ASSERT(res.iterations <= 2, "Jacobi PCG on diagonal matrix must converge in at most 2 iterations.");

  // Check against exact analytic solution x[i] = b[i] / diag[i]
  for (int i = 0; i < n; ++i) {
    double expected = b[i] / diag[i];
    TEST_ASSERT_NEAR(x[i], expected, 1e-7, "Solution mismatch at index " + std::to_string(i));
  }

  double resNorm = computeCpuResidualNorm(mat, x, b);
  TEST_ASSERT(resNorm < 1e-7, "Residual norm should be below 1e-7.");
  return true;
}

// 2. Small SPD Matrix with Known Solution
bool testSmallSpdKnownSolution() {
  // A = [ 4, -1,  0 ]
  //     [-1,  4, -1 ]
  //     [ 0, -1,  4 ]
  // x_exact = [1, 2, 3]^T
  // b = A * x_exact = [4*1 - 2 = 2, -1 + 8 - 3 = 4, -2 + 12 = 10]^T = [2, 4, 10]^T
  int n = 3;
  std::vector<int> rowOffsets = {0, 2, 5, 7};
  std::vector<int> colIndices = {0, 1, 0, 1, 2, 1, 2};
  std::vector<double> values = {4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0};

  CsrMatrix<double> mat(n, n, rowOffsets, colIndices, values);
  std::vector<double> b = {2.0, 4.0, 10.0};
  std::vector<double> expected = {1.0, 2.0, 3.0};

  CudaPcgSolver<double> solver(mat, 1e-9, 100);
  std::vector<double> x(n);
  SolverResult res = solver.solve(b, x);

  TEST_ASSERT(res.hasConverged(), "Solver should converge on small 3x3 SPD matrix.");
  for (int i = 0; i < n; ++i) {
    TEST_ASSERT_NEAR(x[i], expected[i], 1e-6, "Solution mismatch at index " + std::to_string(i));
  }
  return true;
}

// 3. Random Dense/Sparse SPD Matrix with CPU Reference Comparison (Eigen SimplicialLDLT)
bool testRandomSpdCpuReference() {
  int n = 80;
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  // Generate random matrix B and form A = B * B^T + n * I to ensure strong positive-definiteness
  Eigen::MatrixXd B(n, n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      B(i, j) = dist(rng);
    }
  }
  Eigen::MatrixXd denseA = B * B.transpose() + n * Eigen::MatrixXd::Identity(n, n);
  Eigen::SparseMatrix<double> eigenA = denseA.sparseView();

  // Convert Eigen CSC (ColMajor) to CsrMatrix (row-major).
  // This reinterpretation is ONLY correct because A is symmetric (A = A^T),
  // which makes CSC(A) structurally and numerically identical to CSR(A).
  // For non-symmetric matrices, an explicit transpose would be required.
  CsrMatrix<double> mat(n, n, static_cast<int>(eigenA.nonZeros()));
  for (int i = 0; i <= n; ++i) {
    mat.rowOffsets[i] = eigenA.outerIndexPtr()[i];
  }
  for (int i = 0; i < eigenA.nonZeros(); ++i) {
    mat.colIndices[i] = eigenA.innerIndexPtr()[i];
    mat.values[i] = eigenA.valuePtr()[i];
  }
  mat.validate();

  // Generate random RHS b
  std::vector<double> b(n);
  Eigen::VectorXd eigenB(n);
  for (int i = 0; i < n; ++i) {
    b[i] = dist(rng);
    eigenB(i) = b[i];
  }

  // 1. Solve on CPU using Eigen SimplicialLDLT (Trusted CPU reference)
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> cpuSolver(eigenA);
  TEST_ASSERT(cpuSolver.info() == Eigen::Success, "Eigen SimplicialLDLT factorization failed.");
  Eigen::VectorXd cpuX = cpuSolver.solve(eigenB);
  TEST_ASSERT(cpuSolver.info() == Eigen::Success, "Eigen SimplicialLDLT solve failed.");

  // 2. Solve on GPU using CudaPcgSolver
  CudaPcgSolver<double> gpuSolver(mat, 1e-8, 500);
  std::vector<double> gpuX(n);
  SolverResult res = gpuSolver.solve(b, gpuX);

  TEST_ASSERT(res.hasConverged(), "GPU PCG solver failed to converge on random SPD matrix.");

  // 3. Compare GPU solution against CPU reference
  double maxDiff = 0.0;
  for (int i = 0; i < n; ++i) {
    double diff = std::abs(gpuX[i] - cpuX(i));
    maxDiff = std::max(maxDiff, diff);
    TEST_ASSERT_NEAR(gpuX[i], cpuX(i), 1e-4, "GPU and CPU reference differ significantly at index " + std::to_string(i));
  }

  std::cout << "    [INFO] Max difference between GPU and CPU reference: " << maxDiff << std::endl;
  return true;
}

// 4. Different Initial Guesses
bool testDifferentInitialGuesses() {
  int n = 50;
  CsrMatrix<double> mat = CsrMatrix<double>::make1DLaplacian(n, 0.05);

  std::vector<double> b(n, 1.0);

  CudaPcgSolver<double> solver(mat, 1e-8, 1000);

  // Case A: Default zero initial guess x0 = 0
  std::vector<double> xZero(n);
  SolverResult resZero = solver.solve(b, xZero, nullptr);
  TEST_ASSERT(resZero.hasConverged(), "Solve with x0 = 0 should converge.");

  // Case B: Non-zero random initial guess
  std::mt19937 rng(123);
  std::uniform_real_distribution<double> dist(-5.0, 5.0);
  std::vector<double> x0(n);
  for (int i = 0; i < n; ++i) {
    x0[i] = dist(rng);
  }

  std::vector<double> xCustom(n);
  SolverResult resCustom = solver.solve(b, xCustom, &x0);
  TEST_ASSERT(resCustom.hasConverged(), "Solve with non-zero x0 should converge.");

  // Both should reach the exact same unique solution
  for (int i = 0; i < n; ++i) {
    TEST_ASSERT_NEAR(xZero[i], xCustom[i], 1e-5, "Solutions from different x0 must agree at index " + std::to_string(i));
  }
  return true;
}

// 5. Non-Convergence Cases
bool testNonConvergenceCases() {
  int n = 50;
  CsrMatrix<double> mat = CsrMatrix<double>::make1DLaplacian(n, 0.01);
  std::vector<double> b(n, 1.0);

  // Subcase 5A: Insufficient iteration limit (maxIters = 1)
  {
    CudaPcgSolver<double> solver(mat, 1e-8, 1);
    std::vector<double> x(n);
    SolverResult res = solver.solve(b, x);
    TEST_ASSERT(!res.hasConverged(), "Solver must not report convergence with maxIters = 1.");
    TEST_ASSERT(res.status == SolverStatus::MAX_ITERATIONS_REACHED,
                "Status must be MAX_ITERATIONS_REACHED.");
    TEST_ASSERT(res.iterations == 1, "Iterations must equal maxIters.");
  }

  // Subcase 5B: Indefinite / non-SPD matrix (negative diagonal entry)
  {
    // A = [ -2, 1 ]
    //     [  1, 2 ]  (indefinite, eigenvalues has negative curvature)
    std::vector<int> rowOffsets = {0, 2, 4};
    std::vector<int> colIndices = {0, 1, 0, 1};
    std::vector<double> values = {-2.0, 1.0, 1.0, 2.0};
    CsrMatrix<double> indefMat(2, 2, rowOffsets, colIndices, values);

    CudaPcgSolver<double> solver(indefMat, 1e-6, 100);
    std::vector<double> indefB = {1.0, 1.0};
    std::vector<double> x(2);
    SolverResult res = solver.solve(indefB, x);

    // PCG must detect breakdown / non-SPD
    TEST_ASSERT(!res.hasConverged(), "PCG must not succeed on indefinite matrix.");
    TEST_ASSERT(res.status == SolverStatus::BREAKDOWN_NOT_SPD,
                "PCG must detect BREAKDOWN_NOT_SPD on indefinite matrix.");
  }

  return true;
}

// 6. Repeated Solves on Same Matrix (Buffer Persistence & Reusability)
bool testRepeatedSolves() {
  int n = 40;
  CsrMatrix<double> mat = CsrMatrix<double>::make1DLaplacian(n, 0.1);

  CudaPcgSolver<double> solver(mat, 1e-8, 500);

  // Perform 5 consecutive solves with different RHS vectors
  for (int iter = 0; iter < 5; ++iter) {
    std::vector<double> b(n);
    for (int i = 0; i < n; ++i) {
      b[i] = std::sin(iter + 1.0 + i);
    }

    std::vector<double> x(n);
    SolverResult res = solver.solve(b, x);
    TEST_ASSERT(res.hasConverged(), "Repeated solve " + std::to_string(iter) + " failed to converge.");

    double resNorm = computeCpuResidualNorm(mat, x, b);
    TEST_ASSERT(resNorm < 1e-6, "Residual norm too high on repeated solve " + std::to_string(iter));
  }
  return true;
}

// 7. Float Precision Support (CudaPcgSolver<float>)
bool testFloatPrecision() {
  int n = 30;
  CsrMatrix<float> mat = CsrMatrix<float>::make1DLaplacian(n, 0.1f);
  std::vector<float> b(n, 1.0f);

  CudaPcgSolver<float> solver(mat, 1e-5, 200);
  std::vector<float> x(n);
  SolverResult res = solver.solve(b, x);

  TEST_ASSERT(res.hasConverged(), "Float precision solver should converge.");
  double resNorm = computeCpuResidualNorm(mat, x, b);
  TEST_ASSERT(resNorm < 1e-3, "Float residual norm should be below 1e-3.");
  return true;
}

// 8. Zero RHS vector (b = 0)
bool testZeroRhs() {
  int n = 20;
  CsrMatrix<double> mat = CsrMatrix<double>::make1DLaplacian(n, 0.1);
  std::vector<double> b(n, 0.0);

  CudaPcgSolver<double> solver(mat, 1e-8, 100);
  std::vector<double> x(n, 999.0); // deliberately non-zero
  SolverResult res = solver.solve(b, x);

  TEST_ASSERT(res.hasConverged() || res.status == SolverStatus::SUCCESS,
              "Solver should handle zero RHS gracefully.");
  TEST_ASSERT(res.iterations == 0, "Zero RHS should require 0 iterations.");
  for (int i = 0; i < n; ++i) {
    TEST_ASSERT_NEAR(x[i], 0.0, 1e-14, "Solution must be zero for zero RHS at index " + std::to_string(i));
  }
  return true;
}

// 9. RHS Size Mismatch
bool testRhsSizeMismatch() {
  int n = 10;
  CsrMatrix<double> mat = CsrMatrix<double>::make1DLaplacian(n, 0.1);

  CudaPcgSolver<double> solver(mat, 1e-8, 100);

  // b has wrong size
  std::vector<double> bWrong(n + 5, 1.0);
  std::vector<double> x(n);
  SolverResult res = solver.solve(bWrong, x);

  TEST_ASSERT(res.status == SolverStatus::INVALID_INPUT,
              "Solver must return INVALID_INPUT for mismatched b size.");
  TEST_ASSERT(!res.hasConverged(), "Solver must not report convergence with wrong b size.");
  return true;
}

// 10. Matrix Re-initialization (setMatrix called twice)
bool testMatrixReinitialization() {
  int n1 = 20;
  CsrMatrix<double> mat1 = CsrMatrix<double>::make1DLaplacian(n1, 0.1);
  std::vector<double> b1(n1, 1.0);

  CudaPcgSolver<double> solver(mat1, 1e-8, 500);
  std::vector<double> x1(n1);
  SolverResult res1 = solver.solve(b1, x1);
  TEST_ASSERT(res1.hasConverged(), "First matrix solve should converge.");

  // Re-initialize with a different-sized matrix
  int n2 = 30;
  CsrMatrix<double> mat2 = CsrMatrix<double>::make1DLaplacian(n2, 0.2);
  solver.setMatrix(mat2);

  std::vector<double> b2(n2, 2.0);
  std::vector<double> x2(n2);
  SolverResult res2 = solver.solve(b2, x2);
  TEST_ASSERT(res2.hasConverged(), "Second matrix solve should converge after re-initialization.");

  // Verify the solution is actually for the second matrix
  double resNorm = computeCpuResidualNorm(mat2, x2, b2);
  TEST_ASSERT(resNorm < 1e-6, "Residual for second matrix should be small.");
  return true;
}

// 11. throwOnFailure Mode
bool testThrowOnFailure() {
  int n = 50;
  CsrMatrix<double> mat = CsrMatrix<double>::make1DLaplacian(n, 0.01);
  std::vector<double> b(n, 1.0);

  // With throwOnFailure = true and maxIters = 1, should throw
  CudaPcgSolver<double> solver(mat, 1e-12, 1);
  solver.setThrowOnFailure(true);

  std::vector<double> x(n);
  bool threw = false;
  try {
    solver.solve(b, x);
  } catch (const std::runtime_error& e) {
    threw = true;
    std::string msg = e.what();
    TEST_ASSERT(msg.find("failed") != std::string::npos || msg.find("Maximum") != std::string::npos,
                "Exception message should indicate failure.");
  }
  TEST_ASSERT(threw, "Solver must throw when throwOnFailure is enabled and solve fails.");

  // With throwOnFailure = false, should return result without throwing
  solver.setThrowOnFailure(false);
  SolverResult res = solver.solve(b, x);
  TEST_ASSERT(!res.hasConverged(), "Solver should not converge with maxIters = 1.");
  // No exception — test passes if we reach here
  return true;
}

// 12. Large Sparse Matrix (exercises GPU parallelism)
bool testLargeSparseMatrix() {
  int n = 5000;
  CsrMatrix<double> mat = CsrMatrix<double>::make1DLaplacian(n, 0.01);

  std::mt19937 rng(77);
  std::uniform_real_distribution<double> dist(0.1, 2.0);
  std::vector<double> b(n);
  for (int i = 0; i < n; ++i) {
    b[i] = dist(rng);
  }

  CudaPcgSolver<double> solver(mat, 1e-8, 5000);
  std::vector<double> x(n);
  SolverResult res = solver.solve(b, x);

  TEST_ASSERT(res.hasConverged(), "Solver should converge on large 1D Laplacian.");

  double resNorm = computeCpuResidualNorm(mat, x, b);
  double bNorm = 0;
  for (int i = 0; i < n; ++i) bNorm += b[i] * b[i];
  bNorm = std::sqrt(bNorm);
  double relRes = resNorm / bNorm;

  TEST_ASSERT(relRes < 1e-6, "Relative residual on large matrix should be below 1e-6.");
  std::cout << "    [INFO] Large matrix (n=" << n << "): " << res.iterations << " iterations, "
            << "relative residual = " << relRes << ", time = " << res.solveTimeMs << " ms" << std::endl;
  return true;
}

// 13. Ill-Conditioned SPD Matrix
bool testIllConditionedSpd() {
  // Diagonal matrix with condition number ~1e8
  int n = 50;
  std::vector<double> diag(n);
  for (int i = 0; i < n; ++i) {
    diag[i] = 1e-4 + (1e4 - 1e-4) * static_cast<double>(i) / (n - 1);
  }
  CsrMatrix<double> mat = CsrMatrix<double>::fromDiagonal(diag);

  std::vector<double> b(n, 1.0);

  CudaPcgSolver<double> solver(mat, 1e-6, 500);
  std::vector<double> x(n);
  SolverResult res = solver.solve(b, x);

  TEST_ASSERT(res.hasConverged(), "Jacobi-preconditioned PCG on diagonal should converge even if ill-conditioned.");

  // Verify solution: x[i] = b[i] / diag[i] = 1.0 / diag[i]
  for (int i = 0; i < n; ++i) {
    double expected = 1.0 / diag[i];
    TEST_ASSERT_NEAR(x[i], expected, 1e-4 * std::abs(expected),
                     "Solution mismatch at index " + std::to_string(i));
  }
  return true;
}

// =========================================================================
// Main Entrypoint
// =========================================================================

int main() {
  std::cout << "========================================================" << std::endl;
  std::cout << "  CUDA Jacobi-Preconditioned CG Solver Test Suite" << std::endl;
  std::cout << "========================================================" << std::endl;

  RUN_TEST(testDiagonalSpd);
  RUN_TEST(testSmallSpdKnownSolution);
  RUN_TEST(testRandomSpdCpuReference);
  RUN_TEST(testDifferentInitialGuesses);
  RUN_TEST(testNonConvergenceCases);
  RUN_TEST(testRepeatedSolves);
  RUN_TEST(testFloatPrecision);
  RUN_TEST(testZeroRhs);
  RUN_TEST(testRhsSizeMismatch);
  RUN_TEST(testMatrixReinitialization);
  RUN_TEST(testThrowOnFailure);
  RUN_TEST(testLargeSparseMatrix);
  RUN_TEST(testIllConditionedSpd);

  std::cout << "========================================================" << std::endl;
  std::cout << "  Test Summary: " << g_numTestsPassed << "/" << g_numTestsRun << " tests passed." << std::endl;
  if (g_numTestsFailed > 0) {
    std::cout << "  FAILURE: " << g_numTestsFailed << " test(s) failed." << std::endl;
    std::cout << "========================================================" << std::endl;
    return 1;
  }
  std::cout << "  ALL TESTS PASSED SUCCESSFULLY!" << std::endl;
  std::cout << "========================================================" << std::endl;
  return 0;
}

