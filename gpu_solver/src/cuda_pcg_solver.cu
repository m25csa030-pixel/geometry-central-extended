#include "gpu_solver/cuda_pcg_solver.h"
#include "gpu_solver/cuda_utils.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

namespace gpu_solver {

namespace {

// =========================================================================
// CUDA Kernels
// =========================================================================

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

// =========================================================================
// cuBLAS & cuSPARSE Type Traits
// =========================================================================

template <typename T>
struct CudaTraits;

template <>
struct CudaTraits<double> {
  static constexpr cudaDataType_t cudaDataType = CUDA_R_64F;

  static void dot(cublasHandle_t handle, int n, const double* x, int incx, const double* y, int incy, double* result) {
    GPU_SOLVER_CUBLAS_CHECK(cublasDdot(handle, n, x, incx, y, incy, result));
  }

  static void axpy(cublasHandle_t handle, int n, const double* alpha, const double* x, int incx, double* y, int incy) {
    GPU_SOLVER_CUBLAS_CHECK(cublasDaxpy(handle, n, alpha, x, incx, y, incy));
  }

  static void nrm2(cublasHandle_t handle, int n, const double* x, int incx, double* result) {
    GPU_SOLVER_CUBLAS_CHECK(cublasDnrm2(handle, n, x, incx, result));
  }

  static void copy(cublasHandle_t handle, int n, const double* x, int incx, double* y, int incy) {
    GPU_SOLVER_CUBLAS_CHECK(cublasDcopy(handle, n, x, incx, y, incy));
  }
};

template <>
struct CudaTraits<float> {
  static constexpr cudaDataType_t cudaDataType = CUDA_R_32F;

  static void dot(cublasHandle_t handle, int n, const float* x, int incx, const float* y, int incy, float* result) {
    GPU_SOLVER_CUBLAS_CHECK(cublasSdot(handle, n, x, incx, y, incy, result));
  }

  static void axpy(cublasHandle_t handle, int n, const float* alpha, const float* x, int incx, float* y, int incy) {
    GPU_SOLVER_CUBLAS_CHECK(cublasSaxpy(handle, n, alpha, x, incx, y, incy));
  }

  static void nrm2(cublasHandle_t handle, int n, const float* x, int incx, float* result) {
    GPU_SOLVER_CUBLAS_CHECK(cublasSnrm2(handle, n, x, incx, result));
  }

  static void copy(cublasHandle_t handle, int n, const float* x, int incx, float* y, int incy) {
    GPU_SOLVER_CUBLAS_CHECK(cublasScopy(handle, n, x, incx, y, incy));
  }
};

} // namespace

// =========================================================================
// Implementation class (PIMPL)
// =========================================================================

template <typename T>
class CudaPcgSolverImpl {
public:
  double tolerance = 1e-6;
  size_t maxIterations = 1000;
  size_t residualRecomputeInterval = 50; // Recompute true residual r = b - Ax every 50 iters
  bool throwOnFailure = false;

  int nRows = 0;
  int nCols = 0;
  int nnz = 0;
  bool initialized = false;

  // Handles
  cudaStream_t stream = nullptr;
  cublasHandle_t cublasHandle = nullptr;
  cusparseHandle_t cusparseHandle = nullptr;

  // Persistent matrix device buffers
  int* d_rowOffsets = nullptr;
  int* d_colIndices = nullptr;
  T* d_values = nullptr;
  T* d_diagInv = nullptr;

  // Persistent working vector buffers (all size nRows)
  T* d_x = nullptr;
  T* d_b = nullptr;
  T* d_r = nullptr;
  T* d_z = nullptr;
  T* d_p = nullptr;
  T* d_q = nullptr;

  // cuSPARSE descriptors & SpMV buffer
  cusparseSpMatDescr_t matDescr = nullptr;
  cusparseDnVecDescr_t vecPDescr = nullptr;
  cusparseDnVecDescr_t vecQDescr = nullptr;
  cusparseDnVecDescr_t vecXDescr = nullptr;
  void* d_spmvBuffer = nullptr;
  size_t spmvBufferSize = 0;

  // CUDA Events for precise kernel and transfer timing
  cudaEvent_t evStartH2D = nullptr;
  cudaEvent_t evStopH2D = nullptr;
  cudaEvent_t evStartSolve = nullptr;
  cudaEvent_t evStopSolve = nullptr;
  cudaEvent_t evStartD2H = nullptr;
  cudaEvent_t evStopD2H = nullptr;

  std::vector<double> iterationResiduals;

  CudaPcgSolverImpl(double tol, size_t maxIters) : tolerance(tol), maxIterations(maxIters) {
    GPU_SOLVER_CUDA_CHECK(cudaStreamCreate(&stream));
    GPU_SOLVER_CUBLAS_CHECK(cublasCreate(&cublasHandle));
    GPU_SOLVER_CUBLAS_CHECK(cublasSetStream(cublasHandle, stream));
    GPU_SOLVER_CUSPARSE_CHECK(cusparseCreate(&cusparseHandle));
    GPU_SOLVER_CUSPARSE_CHECK(cusparseSetStream(cusparseHandle, stream));

    GPU_SOLVER_CUDA_CHECK(cudaEventCreate(&evStartH2D));
    GPU_SOLVER_CUDA_CHECK(cudaEventCreate(&evStopH2D));
    GPU_SOLVER_CUDA_CHECK(cudaEventCreate(&evStartSolve));
    GPU_SOLVER_CUDA_CHECK(cudaEventCreate(&evStopSolve));
    GPU_SOLVER_CUDA_CHECK(cudaEventCreate(&evStartD2H));
    GPU_SOLVER_CUDA_CHECK(cudaEventCreate(&evStopD2H));
  }

  ~CudaPcgSolverImpl() {
    freeMatrixAndWorkingBuffers();
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

  void freeMatrixAndWorkingBuffers() {
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

    nRows = 0;
    nCols = 0;
    nnz = 0;
    spmvBufferSize = 0;
    initialized = false;
  }

  void setMatrix(int rows, int cols, int numNonZeros, const int* h_rowOffsets, const int* h_colIndices, const T* h_values) {
    if (rows <= 0 || cols <= 0) {
      throw std::invalid_argument("Matrix dimensions must be positive.");
    }
    if (rows != cols) {
      throw std::invalid_argument("PCG solver requires a square matrix (rows == cols).");
    }
    if (numNonZeros < 0) {
      throw std::invalid_argument("Number of non-zeros must be non-negative.");
    }

    freeMatrixAndWorkingBuffers();

    nRows = rows;
    nCols = cols;
    nnz = numNonZeros;

    // 1. Allocate device memory for CSR matrix
    GPU_SOLVER_CUDA_CHECK(cudaMallocAsync(&d_rowOffsets, (nRows + 1) * sizeof(int), stream));
    GPU_SOLVER_CUDA_CHECK(cudaMallocAsync(&d_colIndices, nnz * sizeof(int), stream));
    GPU_SOLVER_CUDA_CHECK(cudaMallocAsync(&d_values, nnz * sizeof(T), stream));

    // 2. Copy matrix data to device
    GPU_SOLVER_CUDA_CHECK(cudaMemcpyAsync(d_rowOffsets, h_rowOffsets, (nRows + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));
    GPU_SOLVER_CUDA_CHECK(cudaMemcpyAsync(d_colIndices, h_colIndices, nnz * sizeof(int), cudaMemcpyHostToDevice, stream));
    GPU_SOLVER_CUDA_CHECK(cudaMemcpyAsync(d_values, h_values, nnz * sizeof(T), cudaMemcpyHostToDevice, stream));

    // 3. Extract diagonal elements for Jacobi preconditioner on host
    std::vector<T> h_diagInv(nRows, static_cast<T>(1));
    for (int i = 0; i < nRows; ++i) {
      bool foundDiag = false;
      for (int j = h_rowOffsets[i]; j < h_rowOffsets[i + 1]; ++j) {
        if (h_colIndices[j] == i) {
          T val = h_values[j];
          if (std::abs(val) > 1e-15) {
            h_diagInv[i] = static_cast<T>(1) / val;
          } else {
            h_diagInv[i] = static_cast<T>(1); // Fallback for near-zero diagonal
          }
          foundDiag = true;
          break;
        }
      }
      if (!foundDiag) {
        h_diagInv[i] = static_cast<T>(1);
      }
    }

    GPU_SOLVER_CUDA_CHECK(cudaMallocAsync(&d_diagInv, nRows * sizeof(T), stream));
    GPU_SOLVER_CUDA_CHECK(cudaMemcpyAsync(d_diagInv, h_diagInv.data(), nRows * sizeof(T), cudaMemcpyHostToDevice, stream));

    // 4. Allocate persistent working vector buffers (all size nRows)
    GPU_SOLVER_CUDA_CHECK(cudaMallocAsync(&d_x, nRows * sizeof(T), stream));
    GPU_SOLVER_CUDA_CHECK(cudaMallocAsync(&d_b, nRows * sizeof(T), stream));
    GPU_SOLVER_CUDA_CHECK(cudaMallocAsync(&d_r, nRows * sizeof(T), stream));
    GPU_SOLVER_CUDA_CHECK(cudaMallocAsync(&d_z, nRows * sizeof(T), stream));
    GPU_SOLVER_CUDA_CHECK(cudaMallocAsync(&d_p, nRows * sizeof(T), stream));
    GPU_SOLVER_CUDA_CHECK(cudaMallocAsync(&d_q, nRows * sizeof(T), stream));

    // 5. Create cuSPARSE matrix and vector descriptors
    GPU_SOLVER_CUSPARSE_CHECK(cusparseCreateCsr(&matDescr, nRows, nCols, nnz, d_rowOffsets, d_colIndices, d_values,
                                                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                                CudaTraits<T>::cudaDataType));

    GPU_SOLVER_CUSPARSE_CHECK(cusparseCreateDnVec(&vecPDescr, nRows, d_p, CudaTraits<T>::cudaDataType));
    GPU_SOLVER_CUSPARSE_CHECK(cusparseCreateDnVec(&vecQDescr, nRows, d_q, CudaTraits<T>::cudaDataType));
    GPU_SOLVER_CUSPARSE_CHECK(cusparseCreateDnVec(&vecXDescr, nRows, d_x, CudaTraits<T>::cudaDataType));

    // 6. Query and allocate buffer for cuSPARSE SpMV
    T alpha = 1;
    T beta = 0;
    GPU_SOLVER_CUSPARSE_CHECK(cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matDescr,
                                                     vecPDescr, &beta, vecQDescr, CudaTraits<T>::cudaDataType,
                                                     CUSPARSE_SPMV_ALG_DEFAULT, &spmvBufferSize));

    if (spmvBufferSize > 0) {
      GPU_SOLVER_CUDA_CHECK(cudaMallocAsync(&d_spmvBuffer, spmvBufferSize, stream));
    }

    GPU_SOLVER_CUDA_CHECK(cudaStreamSynchronize(stream));
    initialized = true;
  }

  SolverResult solve(const T* h_b, T* h_x, const T* h_x0) {
    if (!initialized) {
      SolverResult res;
      res.status = SolverStatus::NOT_INITIALIZED;
      res.message = "Solver has not been initialized with a matrix.";
      if (throwOnFailure) throw std::runtime_error(res.message);
      return res;
    }

    if (h_b == nullptr || h_x == nullptr) {
      SolverResult res;
      res.status = SolverStatus::INVALID_INPUT;
      res.message = "Right-hand side or output solution pointer is null.";
      if (throwOnFailure) throw std::runtime_error(res.message);
      return res;
    }

    iterationResiduals.clear();
    iterationResiduals.reserve(std::min(maxIterations, size_t(1000)));

    auto startTime = std::chrono::high_resolution_clock::now();

    // 1. Copy b to device (H2D transfer)
    GPU_SOLVER_CUDA_CHECK(cudaEventRecord(evStartH2D, stream));
    GPU_SOLVER_CUDA_CHECK(cudaMemcpyAsync(d_b, h_b, nRows * sizeof(T), cudaMemcpyHostToDevice, stream));
    GPU_SOLVER_CUDA_CHECK(cudaEventRecord(evStopH2D, stream));

    // Start solve-only timer
    GPU_SOLVER_CUDA_CHECK(cudaEventRecord(evStartSolve, stream));

    // 2. Compute ||b||_2
    T bNorm = 0;
    CudaTraits<T>::nrm2(cublasHandle, nRows, d_b, 1, &bNorm);
    GPU_SOLVER_CUDA_CHECK(cudaStreamSynchronize(stream));

    if (bNorm == 0) {
      // Trivial system: b = 0 -> x = 0
      GPU_SOLVER_CUDA_CHECK(cudaEventRecord(evStopSolve, stream));
      GPU_SOLVER_CUDA_CHECK(cudaEventRecord(evStartD2H, stream));
      GPU_SOLVER_CUDA_CHECK(cudaMemsetAsync(d_x, 0, nRows * sizeof(T), stream));
      GPU_SOLVER_CUDA_CHECK(cudaMemcpyAsync(h_x, d_x, nRows * sizeof(T), cudaMemcpyDeviceToHost, stream));
      GPU_SOLVER_CUDA_CHECK(cudaEventRecord(evStopD2H, stream));
      GPU_SOLVER_CUDA_CHECK(cudaStreamSynchronize(stream));

      float h2dMs = 0.0f, solveMs = 0.0f, d2hMs = 0.0f;
      cudaEventElapsedTime(&h2dMs, evStartH2D, evStopH2D);
      cudaEventElapsedTime(&solveMs, evStartSolve, evStopSolve);
      cudaEventElapsedTime(&d2hMs, evStartD2H, evStopD2H);

      SolverResult res;
      res.status = SolverStatus::SUCCESS;
      res.iterations = 0;
      res.initialResidualNorm = 0.0;
      res.finalResidualNorm = 0.0;
      res.relativeResidual = 0.0;
      res.solveTimeMs = static_cast<double>(solveMs);
      res.transferTimeMs = static_cast<double>(h2dMs + d2hMs);
      res.totalTimeMs = static_cast<double>(h2dMs + solveMs + d2hMs);
      res.message = "Trivial zero right-hand side, exact solution x = 0.";
      return res;
    }

    // 3. Initialize x and compute initial residual r = b - A*x
    T alphaSpMV = 1;
    T betaSpMV = 0;

    if (h_x0 != nullptr) {
      // Non-zero initial guess
      GPU_SOLVER_CUDA_CHECK(cudaMemcpyAsync(d_x, h_x0, nRows * sizeof(T), cudaMemcpyHostToDevice, stream));

      // q = A * x0
      GPU_SOLVER_CUSPARSE_CHECK(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alphaSpMV, matDescr,
                                             vecXDescr, &betaSpMV, vecQDescr, CudaTraits<T>::cudaDataType,
                                             CUSPARSE_SPMV_ALG_DEFAULT, d_spmvBuffer));

      // r = b
      CudaTraits<T>::copy(cublasHandle, nRows, d_b, 1, d_r, 1);

      // r = r - 1.0 * q
      T negOne = -1;
      CudaTraits<T>::axpy(cublasHandle, nRows, &negOne, d_q, 1, d_r, 1);
    } else {
      // Default zero initial guess: x = 0, r = b
      GPU_SOLVER_CUDA_CHECK(cudaMemsetAsync(d_x, 0, nRows * sizeof(T), stream));
      CudaTraits<T>::copy(cublasHandle, nRows, d_b, 1, d_r, 1);
    }

    // 4. Initial residual norm
    T rNorm = 0;
    CudaTraits<T>::nrm2(cublasHandle, nRows, d_r, 1, &rNorm);
    GPU_SOLVER_CUDA_CHECK(cudaStreamSynchronize(stream));

    double initialResNorm = static_cast<double>(rNorm);
    double relRes = initialResNorm / static_cast<double>(bNorm);
    iterationResiduals.push_back(initialResNorm);

    if (relRes <= tolerance) {
      GPU_SOLVER_CUDA_CHECK(cudaMemcpyAsync(h_x, d_x, nRows * sizeof(T), cudaMemcpyDeviceToHost, stream));
      GPU_SOLVER_CUDA_CHECK(cudaStreamSynchronize(stream));

      auto endTime = std::chrono::high_resolution_clock::now();
      SolverResult res;
      res.status = SolverStatus::CONVERGED;
      res.iterations = 0;
      res.initialResidualNorm = initialResNorm;
      res.finalResidualNorm = initialResNorm;
      res.relativeResidual = relRes;
      res.solveTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
      res.message = "Initial guess already satisfied tolerance threshold.";
      return res;
    }

    // 5. Preconditioning: z = M^-1 * r
    int threadsPerBlock = 256;
    int numBlocks = (nRows + threadsPerBlock - 1) / threadsPerBlock;

    applyJacobiKernel<T><<<numBlocks, threadsPerBlock, 0, stream>>>(nRows, d_r, d_diagInv, d_z);

    // p = z
    CudaTraits<T>::copy(cublasHandle, nRows, d_z, 1, d_p, 1);

    // gamma = r^T * z
    T gamma = 0;
    CudaTraits<T>::dot(cublasHandle, nRows, d_r, 1, d_z, 1, &gamma);
    GPU_SOLVER_CUDA_CHECK(cudaStreamSynchronize(stream));

    if (gamma <= 0 || std::isnan(static_cast<double>(gamma))) {
      SolverResult res;
      res.status = SolverStatus::BREAKDOWN_NOT_SPD;
      res.iterations = 0;
      res.initialResidualNorm = initialResNorm;
      res.finalResidualNorm = initialResNorm;
      res.relativeResidual = relRes;
      res.message = "Initial inner product (r^T * M^-1 * r) <= 0: Matrix or preconditioner is not positive definite.";
      if (throwOnFailure) throw std::runtime_error(res.message);
      return res;
    }

    // 6. PCG Main Iteration Loop
    SolverStatus exitStatus = SolverStatus::MAX_ITERATIONS_REACHED;
    size_t itersCount = 0;

    for (size_t k = 0; k < maxIterations; ++k) {
      itersCount = k + 1;

      // q = A * p
      GPU_SOLVER_CUSPARSE_CHECK(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alphaSpMV, matDescr,
                                             vecPDescr, &betaSpMV, vecQDescr, CudaTraits<T>::cudaDataType,
                                             CUSPARSE_SPMV_ALG_DEFAULT, d_spmvBuffer));

      // denom = p^T * q
      T denom = 0;
      CudaTraits<T>::dot(cublasHandle, nRows, d_p, 1, d_q, 1, &denom);
      GPU_SOLVER_CUDA_CHECK(cudaStreamSynchronize(stream));

      if (denom <= 0 || std::isnan(static_cast<double>(denom))) {
        exitStatus = SolverStatus::BREAKDOWN_NOT_SPD;
        break;
      }

      // alpha = gamma / denom
      T alpha = gamma / denom;

      // x = x + alpha * p
      CudaTraits<T>::axpy(cublasHandle, nRows, &alpha, d_p, 1, d_x, 1);

      // r = r - alpha * q
      T negAlpha = -alpha;
      CudaTraits<T>::axpy(cublasHandle, nRows, &negAlpha, d_q, 1, d_r, 1);

      // Periodic true residual recomputation r = b - A*x to prevent drift
      if (residualRecomputeInterval > 0 && (k + 1) % residualRecomputeInterval == 0) {
        // q = A * x
        GPU_SOLVER_CUSPARSE_CHECK(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alphaSpMV, matDescr,
                                               vecXDescr, &betaSpMV, vecQDescr, CudaTraits<T>::cudaDataType,
                                               CUSPARSE_SPMV_ALG_DEFAULT, d_spmvBuffer));
        // r = b
        CudaTraits<T>::copy(cublasHandle, nRows, d_b, 1, d_r, 1);
        // r = r - 1.0 * q
        T negOne = -1;
        CudaTraits<T>::axpy(cublasHandle, nRows, &negOne, d_q, 1, d_r, 1);
      }

      // Check residual norm
      CudaTraits<T>::nrm2(cublasHandle, nRows, d_r, 1, &rNorm);
      GPU_SOLVER_CUDA_CHECK(cudaStreamSynchronize(stream));

      double currentResNorm = static_cast<double>(rNorm);
      relRes = currentResNorm / static_cast<double>(bNorm);
      iterationResiduals.push_back(currentResNorm);

      // Check for divergence
      if (currentResNorm > 1e10 * initialResNorm || std::isnan(currentResNorm) || std::isinf(currentResNorm)) {
        exitStatus = SolverStatus::NUMERICAL_DIVERGENCE;
        break;
      }

      // Check for convergence
      if (relRes <= tolerance) {
        exitStatus = SolverStatus::CONVERGED;
        break;
      }

      // z = M^-1 * r
      applyJacobiKernel<T><<<numBlocks, threadsPerBlock, 0, stream>>>(nRows, d_r, d_diagInv, d_z);

      // gammaNew = r^T * z
      T gammaNew = 0;
      CudaTraits<T>::dot(cublasHandle, nRows, d_r, 1, d_z, 1, &gammaNew);
      GPU_SOLVER_CUDA_CHECK(cudaStreamSynchronize(stream));

      if (gammaNew <= 0 || std::isnan(static_cast<double>(gammaNew))) {
        exitStatus = SolverStatus::BREAKDOWN_NOT_SPD;
        break;
      }

      T beta = gammaNew / gamma;
      gamma = gammaNew;

      // p = z + beta * p
      updateSearchDirectionKernel<T><<<numBlocks, threadsPerBlock, 0, stream>>>(nRows, d_z, beta, d_p);
    }

    // Stop pure solve timer
    GPU_SOLVER_CUDA_CHECK(cudaEventRecord(evStopSolve, stream));

    // 7. Copy solution back to host (D2H transfer)
    GPU_SOLVER_CUDA_CHECK(cudaEventRecord(evStartD2H, stream));
    GPU_SOLVER_CUDA_CHECK(cudaMemcpyAsync(h_x, d_x, nRows * sizeof(T), cudaMemcpyDeviceToHost, stream));
    GPU_SOLVER_CUDA_CHECK(cudaEventRecord(evStopD2H, stream));
    GPU_SOLVER_CUDA_CHECK(cudaStreamSynchronize(stream));

    float h2dMs = 0.0f, solveMs = 0.0f, d2hMs = 0.0f;
    cudaEventElapsedTime(&h2dMs, evStartH2D, evStopH2D);
    cudaEventElapsedTime(&solveMs, evStartSolve, evStopSolve);
    cudaEventElapsedTime(&d2hMs, evStartD2H, evStopD2H);

    auto endTime = std::chrono::high_resolution_clock::now();
    double elapsedMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    SolverResult res;
    res.status = exitStatus;
    res.iterations = itersCount;
    res.initialResidualNorm = initialResNorm;
    res.finalResidualNorm = static_cast<double>(rNorm);
    res.relativeResidual = relRes;
    res.solveTimeMs = static_cast<double>(solveMs);
    res.transferTimeMs = static_cast<double>(h2dMs + d2hMs);
    res.totalTimeMs = static_cast<double>(h2dMs + solveMs + d2hMs);

    if (exitStatus == SolverStatus::CONVERGED) {
      res.message = "Converged successfully to relative tolerance in " + std::to_string(itersCount) + " iterations.";
    } else if (exitStatus == SolverStatus::MAX_ITERATIONS_REACHED) {
      res.message = "Maximum iteration limit reached without meeting convergence tolerance.";
    } else if (exitStatus == SolverStatus::BREAKDOWN_NOT_SPD) {
      res.message = "PCG breakdown: negative curvature detected (matrix is not positive definite).";
    } else if (exitStatus == SolverStatus::NUMERICAL_DIVERGENCE) {
      res.message = "Numerical divergence: residual norm exceeded bounds or became NaN.";
    }

    if (throwOnFailure && !res.hasConverged()) {
      throw std::runtime_error("CudaPcgSolver failed: " + res.message);
    }

    return res;
  }
};

// =========================================================================
// CudaPcgSolver Public Wrapper Methods
// =========================================================================

template <typename T>
CudaPcgSolver<T>::CudaPcgSolver(double tol, size_t maxIters)
    : pImpl(std::make_unique<CudaPcgSolverImpl<T>>(tol, maxIters)) {}

template <typename T>
CudaPcgSolver<T>::CudaPcgSolver(const CsrMatrix<T>& matrix, double tol, size_t maxIters)
    : pImpl(std::make_unique<CudaPcgSolverImpl<T>>(tol, maxIters)) {
  setMatrix(matrix);
}

template <typename T>
CudaPcgSolver<T>::~CudaPcgSolver() = default;

template <typename T>
CudaPcgSolver<T>::CudaPcgSolver(CudaPcgSolver&&) noexcept = default;

template <typename T>
CudaPcgSolver<T>& CudaPcgSolver<T>::operator=(CudaPcgSolver&&) noexcept = default;

template <typename T>
void CudaPcgSolver<T>::setMatrix(const CsrMatrix<T>& matrix) {
  matrix.validate();
  pImpl->setMatrix(matrix.nRows, matrix.nCols, matrix.nnz, matrix.rowOffsets.data(), matrix.colIndices.data(),
                   matrix.values.data());
}

template <typename T>
void CudaPcgSolver<T>::setMatrix(int nRows, int nCols, int nnz, const int* rowOffsets, const int* colIndices, const T* values) {
  pImpl->setMatrix(nRows, nCols, nnz, rowOffsets, colIndices, values);
}

template <typename T>
SolverResult CudaPcgSolver<T>::solve(const T* h_b, T* h_x, const T* h_x0) {
  return pImpl->solve(h_b, h_x, h_x0);
}

template <typename T>
SolverResult CudaPcgSolver<T>::solve(const std::vector<T>& b, std::vector<T>& x, const std::vector<T>* x0) {
  if (static_cast<int>(b.size()) != pImpl->nRows) {
    SolverResult res;
    res.status = SolverStatus::INVALID_INPUT;
    res.message = "RHS vector b size (" + std::to_string(b.size()) + ") does not match matrix dimension (" +
                  std::to_string(pImpl->nRows) + ").";
    if (pImpl->throwOnFailure) throw std::invalid_argument(res.message);
    return res;
  }
  if (x.size() != static_cast<size_t>(pImpl->nRows)) {
    x.resize(pImpl->nRows);
  }
  const T* x0Ptr = (x0 != nullptr && x0->size() == static_cast<size_t>(pImpl->nRows)) ? x0->data() : nullptr;
  return pImpl->solve(b.data(), x.data(), x0Ptr);
}

template <typename T>
void CudaPcgSolver<T>::setTolerance(double tol) {
  pImpl->tolerance = tol;
}

template <typename T>
double CudaPcgSolver<T>::getTolerance() const {
  return pImpl->tolerance;
}

template <typename T>
void CudaPcgSolver<T>::setMaxIterations(size_t maxIters) {
  pImpl->maxIterations = maxIters;
}

template <typename T>
size_t CudaPcgSolver<T>::getMaxIterations() const {
  return pImpl->maxIterations;
}

template <typename T>
void CudaPcgSolver<T>::setResidualRecomputeInterval(size_t interval) {
  pImpl->residualRecomputeInterval = interval;
}

template <typename T>
size_t CudaPcgSolver<T>::getResidualRecomputeInterval() const {
  return pImpl->residualRecomputeInterval;
}

template <typename T>
void CudaPcgSolver<T>::setThrowOnFailure(bool enable) {
  pImpl->throwOnFailure = enable;
}

template <typename T>
bool CudaPcgSolver<T>::getThrowOnFailure() const {
  return pImpl->throwOnFailure;
}

template <typename T>
bool CudaPcgSolver<T>::isInitialized() const {
  return pImpl->initialized;
}

template <typename T>
size_t CudaPcgSolver<T>::getNumRows() const {
  return pImpl->nRows;
}

template <typename T>
size_t CudaPcgSolver<T>::getNumCols() const {
  return pImpl->nCols;
}

template <typename T>
size_t CudaPcgSolver<T>::getNumNonZeros() const {
  return pImpl->nnz;
}

template <typename T>
const std::vector<double>& CudaPcgSolver<T>::getIterationResiduals() const {
  return pImpl->iterationResiduals;
}

// Explicit template instantiations
template class CudaPcgSolver<double>;
template class CudaPcgSolver<float>;

} // namespace gpu_solver

