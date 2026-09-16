#pragma once

#include "gpu_solver/csr_matrix.h"

#include <memory>
#include <string>
#include <vector>

namespace gpu_solver {

enum class SolverStatus {
  NOT_INITIALIZED,
  SUCCESS,
  CONVERGED,
  MAX_ITERATIONS_REACHED,
  BREAKDOWN_NOT_SPD,
  NUMERICAL_DIVERGENCE,
  INVALID_INPUT
};

inline const char* solverStatusToString(SolverStatus status) {
  switch (status) {
    case SolverStatus::NOT_INITIALIZED:
      return "NOT_INITIALIZED";
    case SolverStatus::SUCCESS:
      return "SUCCESS";
    case SolverStatus::CONVERGED:
      return "CONVERGED";
    case SolverStatus::MAX_ITERATIONS_REACHED:
      return "MAX_ITERATIONS_REACHED";
    case SolverStatus::BREAKDOWN_NOT_SPD:
      return "BREAKDOWN_NOT_SPD";
    case SolverStatus::NUMERICAL_DIVERGENCE:
      return "NUMERICAL_DIVERGENCE";
    case SolverStatus::INVALID_INPUT:
      return "INVALID_INPUT";
    default:
      return "UNKNOWN_STATUS";
  }
}

struct SolverResult {
  SolverStatus status = SolverStatus::NOT_INITIALIZED;
  size_t iterations = 0;
  double initialResidualNorm = 0.0;
  double finalResidualNorm = 0.0;
  double relativeResidual = 0.0;
  double solveTimeMs = 0.0;
  double transferTimeMs = 0.0;
  double totalTimeMs = 0.0;
  std::string message;

  bool hasConverged() const {
    return status == SolverStatus::SUCCESS || status == SolverStatus::CONVERGED;
  }
};

template <typename T>
class CudaPcgSolverImpl;

/**
 * @brief Standalone CUDA Jacobi-preconditioned Conjugate Gradient (PCG) solver.
 *
 * Designed for sparse Symmetric Positive-Definite (SPD) linear systems Ax = b.
 *
 * Key guarantees:
 * - Pre-allocates all working buffers during setMatrix(); zero cudaMalloc inside solve().
 * - Reuses factorized/preconditioned state across multiple solve() invocations.
 * - Supports arbitrary initial guess x0 (or defaults to x0 = 0).
 * - Periodically monitors and tracks convergence history.
 * - Fully RAII encapsulated with PIMPL idiom to insulate public header from CUDA/cuSPARSE headers.
 */
template <typename T>
class CudaPcgSolver {
public:
  explicit CudaPcgSolver(double tol = 1e-6, size_t maxIters = 1000);
  CudaPcgSolver(const CsrMatrix<T>& matrix, double tol = 1e-6, size_t maxIters = 1000);
  ~CudaPcgSolver();

  // Non-copyable, movable
  CudaPcgSolver(const CudaPcgSolver&) = delete;
  CudaPcgSolver& operator=(const CudaPcgSolver&) = delete;
  CudaPcgSolver(CudaPcgSolver&&) noexcept;
  CudaPcgSolver& operator=(CudaPcgSolver&&) noexcept;

  // Matrix setup and preconditioning
  void setMatrix(const CsrMatrix<T>& matrix);
  void setMatrix(int nRows, int nCols, int nnz, const int* rowOffsets, const int* colIndices, const T* values);

  // Solve interface
  SolverResult solve(const T* h_b, T* h_x, const T* h_x0 = nullptr);
  SolverResult solve(const std::vector<T>& b, std::vector<T>& x, const std::vector<T>* x0 = nullptr);

  // Configuration getters and setters
  void setTolerance(double tol);
  double getTolerance() const;

  void setMaxIterations(size_t maxIters);
  size_t getMaxIterations() const;

  void setResidualRecomputeInterval(size_t interval);
  size_t getResidualRecomputeInterval() const;

  void setThrowOnFailure(bool enable);
  bool getThrowOnFailure() const;

  // Diagnostics and inspection
  bool isInitialized() const;
  size_t getNumRows() const;
  size_t getNumCols() const;
  size_t getNumNonZeros() const;
  const std::vector<double>& getIterationResiduals() const;

private:
  std::unique_ptr<CudaPcgSolverImpl<T>> pImpl;
};

// Explicit template instantiations for double and float
extern template class CudaPcgSolver<double>;
extern template class CudaPcgSolver<float>;

} // namespace gpu_solver

