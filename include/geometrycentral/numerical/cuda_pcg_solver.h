#pragma once

#include "geometrycentral/numerical/linear_solvers.h"

#include <memory>

namespace geometrycentral {

template <typename T>
struct CUDAPCGInternals;

/**
 * @brief GPU-accelerated sparse Symmetric Positive-Definite (SPD) linear solver
 * implementing the LinearSolver<T> interface via Jacobi-preconditioned Conjugate Gradient (PCG).
 *
 * Employs cuSPARSE and cuBLAS for high-throughput device execution.
 * Pre-allocates working buffers during construction to ensure zero device allocations inside solve().
 */
template <typename T>
class CUDAPCGPositiveDefiniteSolver final : public LinearSolver<T> {
public:
  CUDAPCGPositiveDefiniteSolver(SparseMatrix<T>& mat, double tol = 1e-6, size_t maxIters = 2000);
  ~CUDAPCGPositiveDefiniteSolver() override;

  // Non-copyable, movable
  CUDAPCGPositiveDefiniteSolver(const CUDAPCGPositiveDefiniteSolver&) = delete;
  CUDAPCGPositiveDefiniteSolver& operator=(const CUDAPCGPositiveDefiniteSolver&) = delete;
  CUDAPCGPositiveDefiniteSolver(CUDAPCGPositiveDefiniteSolver&&) noexcept;
  CUDAPCGPositiveDefiniteSolver& operator=(CUDAPCGPositiveDefiniteSolver&&) noexcept;

  // LinearSolver<T> overrides
  Vector<T> solve(const Vector<T>& rhs) override;
  void solve(Vector<T>& x, const Vector<T>& rhs) override;

  // Configuration and diagnostics
  void setTolerance(double tol);
  double getTolerance() const;

  void setMaxIterations(size_t maxIters);
  size_t getMaxIterations() const;

  size_t getIterationsAchieved() const;
  double getFinalResidual() const;
  double getSolveTimeMs() const;
  double getTransferTimeMs() const;
  double getTotalTimeMs() const;

private:
  std::unique_ptr<CUDAPCGInternals<T>> internals;
};

// Explicit instantiations
extern template class CUDAPCGPositiveDefiniteSolver<double>;
extern template class CUDAPCGPositiveDefiniteSolver<float>;

/**
 * @brief GPU-accelerated sparse Hermitian Positive-Definite (HPD) linear solver
 * implementing LinearSolver<std::complex<double>> by mapping the complex system
 * (A_r + i A_i)(x_r + i x_i) = (b_r + i b_i) into an equivalent 2N x 2N real SPD system:
 * [ A_r  -A_i ] [ x_r ] = [ b_r ]
 * [ A_i   A_r ] [ x_i ]   [ b_i ]
 * and executing via CUDAPCGPositiveDefiniteSolver<double>.
 */
class ComplexCUDAPCGPositiveDefiniteSolver final : public LinearSolver<std::complex<double>> {
public:
  ComplexCUDAPCGPositiveDefiniteSolver(SparseMatrix<std::complex<double>>& mat, double tol = 1e-6,
                                       size_t maxIters = 2000);
  ~ComplexCUDAPCGPositiveDefiniteSolver() override;

  // Non-copyable, movable
  ComplexCUDAPCGPositiveDefiniteSolver(const ComplexCUDAPCGPositiveDefiniteSolver&) = delete;
  ComplexCUDAPCGPositiveDefiniteSolver& operator=(const ComplexCUDAPCGPositiveDefiniteSolver&) = delete;
  ComplexCUDAPCGPositiveDefiniteSolver(ComplexCUDAPCGPositiveDefiniteSolver&&) noexcept;
  ComplexCUDAPCGPositiveDefiniteSolver& operator=(ComplexCUDAPCGPositiveDefiniteSolver&&) noexcept;

  // LinearSolver<std::complex<double>> overrides
  Vector<std::complex<double>> solve(const Vector<std::complex<double>>& rhs) override;
  void solve(Vector<std::complex<double>>& x, const Vector<std::complex<double>>& rhs) override;

  // Configuration and diagnostics
  void setTolerance(double tol);
  double getTolerance() const;

  void setMaxIterations(size_t maxIters);
  size_t getMaxIterations() const;

  size_t getIterationsAchieved() const;
  double getFinalResidual() const;
  double getSolveTimeMs() const;
  double getTransferTimeMs() const;
  double getTotalTimeMs() const;

private:
  std::unique_ptr<CUDAPCGPositiveDefiniteSolver<double>> realSolver;
  size_t nComplex;
};

#ifdef GC_HAVE_CUDA
template <typename T>
Vector<T> solvePositiveDefiniteCUDA(SparseMatrix<T>& matrix, const Vector<T>& rhs, double tol = 1e-6,
                                    size_t maxIters = 2000);
#endif

} // namespace geometrycentral

