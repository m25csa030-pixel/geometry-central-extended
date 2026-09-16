#include "geometrycentral/numerical/cuda_pcg_solver.h"
#include "geometrycentral/numerical/linear_algebra_utilities.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace geometrycentral {

namespace {

#define GC_CUDA_CHECK(call)                                                                                        \
  do {                                                                                                             \
    cudaError_t err = (call);                                                                                      \
    if (err != cudaSuccess) {                                                                                      \
      throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err) + " at " + __FILE__ + ":" +  \
                               std::to_string(__LINE__));                                                          \
    }                                                                                                              \
  } while (0)

#define GC_CUSPARSE_CHECK(call)                                                                                        \
  do {                                                                                                                 \
    cusparseStatus_t status = (call);                                                                                  \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                                                           \
      throw std::runtime_error(std::string("cuSPARSE Error: ") + cusparseGetErrorString(status) + " at " + __FILE__ + \
                               ":" + std::to_string(__LINE__));                                                        \
    }                                                                                                                  \
  } while (0)

#define GC_CUBLAS_CHECK(call)                                                                                     \
  do {                                                                                                            \
    cublasStatus_t status = (call);                                                                               \
    if (status != CUBLAS_STATUS_SUCCESS) {                                                                        \
      throw std::runtime_error(std::string("cuBLAS Error (status ") + std::to_string(status) + ") at " + __FILE__ + \
                               ":" + std::to_string(__LINE__));                                                   \
    }                                                                                                             \
  } while (0)

// CUDA Kernels
template <typename T>
__global__ void applyJacobiKernel(int n, const T* __restrict__ r, const T* __restrict__ dinv, T* __restrict__ z) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    z[i] = r[i] * dinv[i];
  }
}

template <typename T>
__global__ void updateSearchDirectionKernel(int n, const T* __restrict__ z, T beta, T* __restrict__ p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    p[i] = z[i] + beta * p[i];
  }
}

// cuBLAS & cuSPARSE traits
template <typename T>
struct CudaTraits;

template <>
struct CudaTraits<double> {
  static constexpr cudaDataType_t cudaDataType = CUDA_R_64F;

  static void dot(cublasHandle_t handle, int n, const double* x, int incx, const double* y, int incy, double* result) {
    GC_CUBLAS_CHECK(cublasDdot(handle, n, x, incx, y, incy, result));
  }
  static void axpy(cublasHandle_t handle, int n, const double* alpha, const double* x, int incx, double* y, int incy) {
    GC_CUBLAS_CHECK(cublasDaxpy(handle, n, alpha, x, incx, y, incy));
  }
  static void nrm2(cublasHandle_t handle, int n, const double* x, int incx, double* result) {
    GC_CUBLAS_CHECK(cublasDnrm2(handle, n, x, incx, result));
  }
  static void copy(cublasHandle_t handle, int n, const double* x, int incx, double* y, int incy) {
    GC_CUBLAS_CHECK(cublasDcopy(handle, n, x, incx, y, incy));
  }
};

template <>
struct CudaTraits<float> {
  static constexpr cudaDataType_t cudaDataType = CUDA_R_32F;

  static void dot(cublasHandle_t handle, int n, const float* x, int incx, const float* y, int incy, float* result) {
    GC_CUBLAS_CHECK(cublasSdot(handle, n, x, incx, y, incy, result));
  }
  static void axpy(cublasHandle_t handle, int n, const float* alpha, const float* x, int incx, float* y, int incy) {
    GC_CUBLAS_CHECK(cublasSaxpy(handle, n, alpha, x, incx, y, incy));
  }
  static void nrm2(cublasHandle_t handle, int n, const float* x, int incx, float* result) {
    GC_CUBLAS_CHECK(cublasSnrm2(handle, n, x, incx, result));
  }
  static void copy(cublasHandle_t handle, int n, const float* x, int incx, float* y, int incy) {
    GC_CUBLAS_CHECK(cublasScopy(handle, n, x, incx, y, incy));
  }
};

} // namespace

template <typename T>
struct CUDAPCGInternals {
  double tolerance = 1e-6;
  size_t maxIterations = 2000;

  int nRows = 0;
  int nCols = 0;
  int nnz = 0;

  cudaStream_t stream = nullptr;
  cublasHandle_t cublasHandle = nullptr;
  cusparseHandle_t cusparseHandle = nullptr;

  // Device matrix buffers
  int* d_rowOffsets = nullptr;
  int* d_colIndices = nullptr;
  T* d_values = nullptr;
  T* d_diagInv = nullptr;

  // Persistent working vectors
  T* d_x = nullptr;
  T* d_b = nullptr;
  T* d_r = nullptr;
  T* d_z = nullptr;
  T* d_p = nullptr;
  T* d_q = nullptr;

  // cuSPARSE descriptors
  cusparseSpMatDescr_t matDescr = nullptr;
  cusparseDnVecDescr_t vecPDescr = nullptr;
  cusparseDnVecDescr_t vecQDescr = nullptr;
  cusparseDnVecDescr_t vecXDescr = nullptr;
  void* d_spmvBuffer = nullptr;
  size_t spmvBufferSize = 0;

  // Timing events
  cudaEvent_t evStartH2D = nullptr;
  cudaEvent_t evStopH2D = nullptr;
  cudaEvent_t evStartSolve = nullptr;
  cudaEvent_t evStopSolve = nullptr;
  cudaEvent_t evStartD2H = nullptr;
  cudaEvent_t evStopD2H = nullptr;

  // Diagnostics
  size_t iterationsAchieved = 0;
  double finalResidual = 0.0;
  double solveTimeMs = 0.0;
  double transferTimeMs = 0.0;
  double totalTimeMs = 0.0;

  CUDAPCGInternals(double tol, size_t maxIters) : tolerance(tol), maxIterations(maxIters) {
    GC_CUDA_CHECK(cudaStreamCreate(&stream));
    GC_CUBLAS_CHECK(cublasCreate(&cublasHandle));
    GC_CUBLAS_CHECK(cublasSetStream(cublasHandle, stream));
    GC_CUSPARSE_CHECK(cusparseCreate(&cusparseHandle));
    GC_CUSPARSE_CHECK(cusparseSetStream(cusparseHandle, stream));

    GC_CUDA_CHECK(cudaEventCreate(&evStartH2D));
    GC_CUDA_CHECK(cudaEventCreate(&evStopH2D));
    GC_CUDA_CHECK(cudaEventCreate(&evStartSolve));
    GC_CUDA_CHECK(cudaEventCreate(&evStopSolve));
    GC_CUDA_CHECK(cudaEventCreate(&evStartD2H));
    GC_CUDA_CHECK(cudaEventCreate(&evStopD2H));
  }

  ~CUDAPCGInternals() {
    freeBuffers();
    if (evStartH2D) cudaEventDestroy(evStartH2D);
    if (evStopH2D) cudaEventDestroy(evStopH2D);
    if (evStartSolve) cudaEventDestroy(evStartSolve);
    if (evStopSolve) cudaEventDestroy(evStopSolve);
    if (evStartD2H) cudaEventDestroy(evStartD2H);
    if (evStopD2H) cudaEventDestroy(evStopD2H);
    if (cusparseHandle) cusparseDestroy(cusparseHandle);
    if (cublasHandle) cublasDestroy(cublasHandle);
    if (stream) cudaStreamDestroy(stream);
  }

  void freeBuffers() {
    if (matDescr) {
      cusparseDestroySpMat(matDescr);
      matDescr = nullptr;
    }
    if (vecPDescr) {
      cusparseDestroyDnVec(vecPDescr);
      vecPDescr = nullptr;
    }
    if (vecQDescr) {
      cusparseDestroyDnVec(vecQDescr);
      vecQDescr = nullptr;
    }
    if (vecXDescr) {
      cusparseDestroyDnVec(vecXDescr);
      vecXDescr = nullptr;
    }

    if (d_spmvBuffer) {
      cudaFree(d_spmvBuffer);
      d_spmvBuffer = nullptr;
    }
    if (d_rowOffsets) {
      cudaFree(d_rowOffsets);
      d_rowOffsets = nullptr;
    }
    if (d_colIndices) {
      cudaFree(d_colIndices);
      d_colIndices = nullptr;
    }
    if (d_values) {
      cudaFree(d_values);
      d_values = nullptr;
    }
    if (d_diagInv) {
      cudaFree(d_diagInv);
      d_diagInv = nullptr;
    }
    if (d_x) {
      cudaFree(d_x);
      d_x = nullptr;
    }
    if (d_b) {
      cudaFree(d_b);
      d_b = nullptr;
    }
    if (d_r) {
      cudaFree(d_r);
      d_r = nullptr;
    }
    if (d_z) {
      cudaFree(d_z);
      d_z = nullptr;
    }
    if (d_p) {
      cudaFree(d_p);
      d_p = nullptr;
    }
    if (d_q) {
      cudaFree(d_q);
      d_q = nullptr;
    }
  }

  void setMatrix(int rows, int cols, int numNonZeros, const int* h_rowOffsets, const int* h_colIndices,
                 const T* h_values) {
    freeBuffers();

    nRows = rows;
    nCols = cols;
    nnz = numNonZeros;

    // Allocate CSR matrix on device
    GC_CUDA_CHECK(cudaMallocAsync(&d_rowOffsets, (nRows + 1) * sizeof(int), stream));
    GC_CUDA_CHECK(cudaMallocAsync(&d_colIndices, nnz * sizeof(int), stream));
    GC_CUDA_CHECK(cudaMallocAsync(&d_values, nnz * sizeof(T), stream));

    GC_CUDA_CHECK(cudaMemcpyAsync(d_rowOffsets, h_rowOffsets, (nRows + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));
    GC_CUDA_CHECK(cudaMemcpyAsync(d_colIndices, h_colIndices, nnz * sizeof(int), cudaMemcpyHostToDevice, stream));
    GC_CUDA_CHECK(cudaMemcpyAsync(d_values, h_values, nnz * sizeof(T), cudaMemcpyHostToDevice, stream));

    // Extract diagonal elements for Jacobi preconditioner on host
    std::vector<T> h_diagInv(nRows, static_cast<T>(1));
    for (int i = 0; i < nRows; ++i) {
      bool foundDiag = false;
      for (int j = h_rowOffsets[i]; j < h_rowOffsets[i + 1]; ++j) {
        if (h_colIndices[j] == i) {
          T val = h_values[j];
          if (std::abs(val) > 1e-15) {
            h_diagInv[i] = static_cast<T>(1) / val;
          } else {
            h_diagInv[i] = static_cast<T>(1);
          }
          foundDiag = true;
          break;
        }
      }
      if (!foundDiag) {
        h_diagInv[i] = static_cast<T>(1);
      }
    }

    GC_CUDA_CHECK(cudaMallocAsync(&d_diagInv, nRows * sizeof(T), stream));
    GC_CUDA_CHECK(cudaMemcpyAsync(d_diagInv, h_diagInv.data(), nRows * sizeof(T), cudaMemcpyHostToDevice, stream));

    // Pre-allocate working vectors
    GC_CUDA_CHECK(cudaMallocAsync(&d_x, nRows * sizeof(T), stream));
    GC_CUDA_CHECK(cudaMallocAsync(&d_b, nRows * sizeof(T), stream));
    GC_CUDA_CHECK(cudaMallocAsync(&d_r, nRows * sizeof(T), stream));
    GC_CUDA_CHECK(cudaMallocAsync(&d_z, nRows * sizeof(T), stream));
    GC_CUDA_CHECK(cudaMallocAsync(&d_p, nRows * sizeof(T), stream));
    GC_CUDA_CHECK(cudaMallocAsync(&d_q, nRows * sizeof(T), stream));

    // Descriptors
    GC_CUSPARSE_CHECK(cusparseCreateCsr(&matDescr, nRows, nCols, nnz, d_rowOffsets, d_colIndices, d_values,
                                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                        CudaTraits<T>::cudaDataType));
    GC_CUSPARSE_CHECK(cusparseCreateDnVec(&vecPDescr, nRows, d_p, CudaTraits<T>::cudaDataType));
    GC_CUSPARSE_CHECK(cusparseCreateDnVec(&vecQDescr, nRows, d_q, CudaTraits<T>::cudaDataType));
    GC_CUSPARSE_CHECK(cusparseCreateDnVec(&vecXDescr, nRows, d_x, CudaTraits<T>::cudaDataType));

    T alpha = 1, beta = 0;
    GC_CUSPARSE_CHECK(cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matDescr,
                                             vecPDescr, &beta, vecQDescr, CudaTraits<T>::cudaDataType,
                                             CUSPARSE_SPMV_ALG_DEFAULT, &spmvBufferSize));
    if (spmvBufferSize > 0) {
      GC_CUDA_CHECK(cudaMallocAsync(&d_spmvBuffer, spmvBufferSize, stream));
    }

    GC_CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void solve(const T* h_b, T* h_x) {
    GC_CUDA_CHECK(cudaEventRecord(evStartH2D, stream));
    GC_CUDA_CHECK(cudaMemcpyAsync(d_b, h_b, nRows * sizeof(T), cudaMemcpyHostToDevice, stream));
    GC_CUDA_CHECK(cudaEventRecord(evStopH2D, stream));

    GC_CUDA_CHECK(cudaEventRecord(evStartSolve, stream));

    T bNorm = 0;
    CudaTraits<T>::nrm2(cublasHandle, nRows, d_b, 1, &bNorm);
    GC_CUDA_CHECK(cudaStreamSynchronize(stream));

    if (bNorm == 0) {
      GC_CUDA_CHECK(cudaEventRecord(evStopSolve, stream));
      GC_CUDA_CHECK(cudaEventRecord(evStartD2H, stream));
      GC_CUDA_CHECK(cudaMemsetAsync(d_x, 0, nRows * sizeof(T), stream));
      GC_CUDA_CHECK(cudaMemcpyAsync(h_x, d_x, nRows * sizeof(T), cudaMemcpyDeviceToHost, stream));
      GC_CUDA_CHECK(cudaEventRecord(evStopD2H, stream));
      GC_CUDA_CHECK(cudaStreamSynchronize(stream));

      iterationsAchieved = 0;
      finalResidual = 0.0;
      return;
    }

    // Default zero initial guess: x0 = 0 -> r0 = b
    GC_CUDA_CHECK(cudaMemsetAsync(d_x, 0, nRows * sizeof(T), stream));
    CudaTraits<T>::copy(cublasHandle, nRows, d_b, 1, d_r, 1);

    T rNorm = 0;
    CudaTraits<T>::nrm2(cublasHandle, nRows, d_r, 1, &rNorm);
    GC_CUDA_CHECK(cudaStreamSynchronize(stream));

    double initialResNorm = static_cast<double>(rNorm);
    double relRes = initialResNorm / static_cast<double>(bNorm);

    if (relRes <= tolerance) {
      GC_CUDA_CHECK(cudaEventRecord(evStopSolve, stream));
      GC_CUDA_CHECK(cudaEventRecord(evStartD2H, stream));
      GC_CUDA_CHECK(cudaMemcpyAsync(h_x, d_x, nRows * sizeof(T), cudaMemcpyDeviceToHost, stream));
      GC_CUDA_CHECK(cudaEventRecord(evStopD2H, stream));
      GC_CUDA_CHECK(cudaStreamSynchronize(stream));

      iterationsAchieved = 0;
      finalResidual = relRes;
      return;
    }

    int threadsPerBlock = 256;
    int numBlocks = (nRows + threadsPerBlock - 1) / threadsPerBlock;

    // z0 = M^-1 * r0
    applyJacobiKernel<T><<<numBlocks, threadsPerBlock, 0, stream>>>(nRows, d_r, d_diagInv, d_z);
    CudaTraits<T>::copy(cublasHandle, nRows, d_z, 1, d_p, 1);

    T gamma = 0;
    CudaTraits<T>::dot(cublasHandle, nRows, d_r, 1, d_z, 1, &gamma);
    GC_CUDA_CHECK(cudaStreamSynchronize(stream));

    if (gamma <= 0 || std::isnan(static_cast<double>(gamma))) {
      throw std::runtime_error("CUDAPCGPositiveDefiniteSolver: Initial inner product <= 0 (matrix is not SPD).");
    }

    T alphaSpMV = 1, betaSpMV = 0;
    size_t k = 0;
    for (k = 0; k < maxIterations; ++k) {
      // q = A * p
      GC_CUSPARSE_CHECK(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alphaSpMV, matDescr,
                                     vecPDescr, &betaSpMV, vecQDescr, CudaTraits<T>::cudaDataType,
                                     CUSPARSE_SPMV_ALG_DEFAULT, d_spmvBuffer));

      // denom = p^T * q
      T denom = 0;
      CudaTraits<T>::dot(cublasHandle, nRows, d_p, 1, d_q, 1, &denom);
      GC_CUDA_CHECK(cudaStreamSynchronize(stream));

      if (denom <= 0 || std::isnan(static_cast<double>(denom))) {
        throw std::runtime_error("CUDAPCGPositiveDefiniteSolver breakdown: p^T * A * p <= 0 (matrix not SPD).");
      }

      T alpha = gamma / denom;
      CudaTraits<T>::axpy(cublasHandle, nRows, &alpha, d_p, 1, d_x, 1);

      T negAlpha = -alpha;
      CudaTraits<T>::axpy(cublasHandle, nRows, &negAlpha, d_q, 1, d_r, 1);

      CudaTraits<T>::nrm2(cublasHandle, nRows, d_r, 1, &rNorm);
      GC_CUDA_CHECK(cudaStreamSynchronize(stream));

      relRes = static_cast<double>(rNorm) / static_cast<double>(bNorm);
      if (relRes <= tolerance) {
        k++;
        break;
      }

      // z = M^-1 * r
      applyJacobiKernel<T><<<numBlocks, threadsPerBlock, 0, stream>>>(nRows, d_r, d_diagInv, d_z);

      T gammaNew = 0;
      CudaTraits<T>::dot(cublasHandle, nRows, d_r, 1, d_z, 1, &gammaNew);
      GC_CUDA_CHECK(cudaStreamSynchronize(stream));

      if (gammaNew <= 0 || std::isnan(static_cast<double>(gammaNew))) {
        throw std::runtime_error("CUDAPCGPositiveDefiniteSolver breakdown: r^T * M^-1 * r <= 0.");
      }

      T beta = gammaNew / gamma;
      gamma = gammaNew;

      updateSearchDirectionKernel<T><<<numBlocks, threadsPerBlock, 0, stream>>>(nRows, d_z, beta, d_p);
    }

    GC_CUDA_CHECK(cudaEventRecord(evStopSolve, stream));

    GC_CUDA_CHECK(cudaEventRecord(evStartD2H, stream));
    GC_CUDA_CHECK(cudaMemcpyAsync(h_x, d_x, nRows * sizeof(T), cudaMemcpyDeviceToHost, stream));
    GC_CUDA_CHECK(cudaEventRecord(evStopD2H, stream));
    GC_CUDA_CHECK(cudaStreamSynchronize(stream));

    float h2dMs = 0.0f, solveMs = 0.0f, d2hMs = 0.0f;
    cudaEventElapsedTime(&h2dMs, evStartH2D, evStopH2D);
    cudaEventElapsedTime(&solveMs, evStartSolve, evStopSolve);
    cudaEventElapsedTime(&d2hMs, evStartD2H, evStopD2H);

    iterationsAchieved = k;
    finalResidual = relRes;
    solveTimeMs = static_cast<double>(solveMs);
    transferTimeMs = static_cast<double>(h2dMs + d2hMs);
    totalTimeMs = static_cast<double>(h2dMs + solveMs + d2hMs);
  }
};

// =========================================================================
// Public Class Implementation
// =========================================================================

template <typename T>
CUDAPCGPositiveDefiniteSolver<T>::CUDAPCGPositiveDefiniteSolver(SparseMatrix<T>& mat, double tol, size_t maxIters)
    : LinearSolver<T>(mat), internals(new CUDAPCGInternals<T>(tol, maxIters)) {

  if (this->nRows != this->nCols) {
    throw std::logic_error("Matrix must be square");
  }

#ifndef GC_NLINALG_DEBUG
  checkFinite(mat);
  checkHermitian(mat);
#endif

  mat.makeCompressed();

  internals->setMatrix(static_cast<int>(mat.rows()), static_cast<int>(mat.cols()), static_cast<int>(mat.nonZeros()),
                       mat.outerIndexPtr(), mat.innerIndexPtr(), mat.valuePtr());
}

template <typename T>
CUDAPCGPositiveDefiniteSolver<T>::~CUDAPCGPositiveDefiniteSolver() = default;

template <typename T>
CUDAPCGPositiveDefiniteSolver<T>::CUDAPCGPositiveDefiniteSolver(CUDAPCGPositiveDefiniteSolver&&) noexcept = default;

template <typename T>
CUDAPCGPositiveDefiniteSolver<T>&
CUDAPCGPositiveDefiniteSolver<T>::operator=(CUDAPCGPositiveDefiniteSolver&&) noexcept = default;

template <typename T>
Vector<T> CUDAPCGPositiveDefiniteSolver<T>::solve(const Vector<T>& rhs) {
  Vector<T> out(this->nRows);
  solve(out, rhs);
  return out;
}

template <typename T>
void CUDAPCGPositiveDefiniteSolver<T>::solve(Vector<T>& x, const Vector<T>& rhs) {
  if (static_cast<size_t>(rhs.rows()) != this->nRows) {
    throw std::logic_error("Vector is not the right length");
  }
#ifndef GC_NLINALG_DEBUG
  checkFinite(rhs);
#endif

  if (static_cast<size_t>(x.rows()) != this->nRows) {
    x.resize(this->nRows);
  }

  internals->solve(rhs.data(), x.data());
}

template <typename T>
void CUDAPCGPositiveDefiniteSolver<T>::setTolerance(double tol) {
  internals->tolerance = tol;
}

template <typename T>
double CUDAPCGPositiveDefiniteSolver<T>::getTolerance() const {
  return internals->tolerance;
}

template <typename T>
void CUDAPCGPositiveDefiniteSolver<T>::setMaxIterations(size_t maxIters) {
  internals->maxIterations = maxIters;
}

template <typename T>
size_t CUDAPCGPositiveDefiniteSolver<T>::getMaxIterations() const {
  return internals->maxIterations;
}

template <typename T>
size_t CUDAPCGPositiveDefiniteSolver<T>::getIterationsAchieved() const {
  return internals->iterationsAchieved;
}

template <typename T>
double CUDAPCGPositiveDefiniteSolver<T>::getFinalResidual() const {
  return internals->finalResidual;
}

template <typename T>
double CUDAPCGPositiveDefiniteSolver<T>::getSolveTimeMs() const {
  return internals->solveTimeMs;
}

template <typename T>
double CUDAPCGPositiveDefiniteSolver<T>::getTransferTimeMs() const {
  return internals->transferTimeMs;
}

template <typename T>
double CUDAPCGPositiveDefiniteSolver<T>::getTotalTimeMs() const {
  return internals->totalTimeMs;
}

// =========================================================================
// ComplexCUDAPCGPositiveDefiniteSolver Implementation
// =========================================================================

ComplexCUDAPCGPositiveDefiniteSolver::ComplexCUDAPCGPositiveDefiniteSolver(SparseMatrix<std::complex<double>>& mat,
                                                                           double tol, size_t maxIters)
    : LinearSolver<std::complex<double>>(mat), nComplex(mat.rows()) {
  if (mat.rows() != mat.cols()) {
    throw std::invalid_argument("ComplexCUDAPCGPositiveDefiniteSolver: matrix must be square.");
  }

  // Build 2N x 2N real symmetric system:
  // [  A_r   -A_i ]
  // [  A_i    A_r ]
  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(4 * mat.nonZeros());

  for (int k = 0; k < mat.outerSize(); ++k) {
    for (typename SparseMatrix<std::complex<double>>::InnerIterator it(mat, k); it; ++it) {
      size_t r = it.row();
      size_t c = it.col();
      std::complex<double> v = it.value();
      double vr = v.real();
      double vi = v.imag();

      // Block (0,0): A_r and Block (1,1): A_r
      if (vr != 0.0 || r == c) {
        triplets.emplace_back(r, c, vr);
        triplets.emplace_back(r + nComplex, c + nComplex, vr);
      }
      // Block (0,1): -A_i and Block (1,0): A_i
      if (vi != 0.0) {
        triplets.emplace_back(r, c + nComplex, -vi);
        triplets.emplace_back(r + nComplex, c, vi);
      }
    }
  }

  SparseMatrix<double> realMat(2 * nComplex, 2 * nComplex);
  realMat.setFromTriplets(triplets.begin(), triplets.end());
  realMat.makeCompressed();

  realSolver.reset(new CUDAPCGPositiveDefiniteSolver<double>(realMat, tol, maxIters));
}

ComplexCUDAPCGPositiveDefiniteSolver::~ComplexCUDAPCGPositiveDefiniteSolver() = default;
ComplexCUDAPCGPositiveDefiniteSolver::ComplexCUDAPCGPositiveDefiniteSolver(ComplexCUDAPCGPositiveDefiniteSolver&&) noexcept = default;
ComplexCUDAPCGPositiveDefiniteSolver& ComplexCUDAPCGPositiveDefiniteSolver::operator=(ComplexCUDAPCGPositiveDefiniteSolver&&) noexcept = default;

Vector<std::complex<double>> ComplexCUDAPCGPositiveDefiniteSolver::solve(const Vector<std::complex<double>>& rhs) {
  Vector<std::complex<double>> x(nComplex);
  solve(x, rhs);
  return x;
}

void ComplexCUDAPCGPositiveDefiniteSolver::solve(Vector<std::complex<double>>& x,
                                                 const Vector<std::complex<double>>& rhs) {
  if (static_cast<size_t>(rhs.size()) != nComplex) {
    throw std::invalid_argument("ComplexCUDAPCGPositiveDefiniteSolver: RHS size mismatch.");
  }
  if (static_cast<size_t>(x.size()) != nComplex) {
    x = Vector<std::complex<double>>::Zero(nComplex);
  }

  Vector<double> realRhs(2 * nComplex);
  Vector<double> realX(2 * nComplex);
  for (size_t i = 0; i < nComplex; ++i) {
    realRhs[i] = rhs[i].real();
    realRhs[nComplex + i] = rhs[i].imag();
    realX[i] = x[i].real();
    realX[nComplex + i] = x[i].imag();
  }

  realSolver->solve(realX, realRhs);

  for (size_t i = 0; i < nComplex; ++i) {
    x[i] = std::complex<double>(realX[i], realX[nComplex + i]);
  }
}

void ComplexCUDAPCGPositiveDefiniteSolver::setTolerance(double tol) { realSolver->setTolerance(tol); }
double ComplexCUDAPCGPositiveDefiniteSolver::getTolerance() const { return realSolver->getTolerance(); }
void ComplexCUDAPCGPositiveDefiniteSolver::setMaxIterations(size_t maxIters) { realSolver->setMaxIterations(maxIters); }
size_t ComplexCUDAPCGPositiveDefiniteSolver::getMaxIterations() const { return realSolver->getMaxIterations(); }
size_t ComplexCUDAPCGPositiveDefiniteSolver::getIterationsAchieved() const { return realSolver->getIterationsAchieved(); }
double ComplexCUDAPCGPositiveDefiniteSolver::getFinalResidual() const { return realSolver->getFinalResidual(); }
double ComplexCUDAPCGPositiveDefiniteSolver::getSolveTimeMs() const { return realSolver->getSolveTimeMs(); }
double ComplexCUDAPCGPositiveDefiniteSolver::getTransferTimeMs() const { return realSolver->getTransferTimeMs(); }
double ComplexCUDAPCGPositiveDefiniteSolver::getTotalTimeMs() const { return realSolver->getTotalTimeMs(); }


#ifdef GC_HAVE_CUDA
template <typename T>
Vector<T> solvePositiveDefiniteCUDA(SparseMatrix<T>& matrix, const Vector<T>& rhs, double tol, size_t maxIters) {
  CUDAPCGPositiveDefiniteSolver<T> solver(matrix, tol, maxIters);
  return solver.solve(rhs);
}
#endif

// Explicit template instantiations
template class CUDAPCGPositiveDefiniteSolver<double>;
template class CUDAPCGPositiveDefiniteSolver<float>;

#ifdef GC_HAVE_CUDA
template Vector<double> solvePositiveDefiniteCUDA<double>(SparseMatrix<double>& matrix, const Vector<double>& rhs,
                                                         double tol, size_t maxIters);
template Vector<float> solvePositiveDefiniteCUDA<float>(SparseMatrix<float>& matrix, const Vector<float>& rhs,
                                                       double tol, size_t maxIters);
#endif

} // namespace geometrycentral

