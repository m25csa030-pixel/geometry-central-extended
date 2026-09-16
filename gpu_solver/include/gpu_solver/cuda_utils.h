#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include <stdexcept>
#include <string>

namespace gpu_solver {

inline const char* cublasGetErrorEnum(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
    default:
      return "<unknown cublasStatus_t>";
  }
}

#define GPU_SOLVER_CUDA_CHECK(call)                                                                               \
  do {                                                                                                            \
    cudaError_t err = (call);                                                                                     \
    if (err != cudaSuccess) {                                                                                     \
      throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err) + " at " + __FILE__ + ":" + \
                               std::to_string(__LINE__));                                                         \
    }                                                                                                             \
  } while (0)

#define GPU_SOLVER_CUSPARSE_CHECK(call)                                                                                \
  do {                                                                                                                 \
    cusparseStatus_t status = (call);                                                                                  \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                                                           \
      throw std::runtime_error(std::string("cuSPARSE Error: ") + cusparseGetErrorString(status) + " at " + __FILE__ + \
                               ":" + std::to_string(__LINE__));                                                        \
    }                                                                                                                  \
  } while (0)

#define GPU_SOLVER_CUBLAS_CHECK(call)                                                                                  \
  do {                                                                                                                 \
    cublasStatus_t status = (call);                                                                                    \
    if (status != CUBLAS_STATUS_SUCCESS) {                                                                             \
      throw std::runtime_error(std::string("cuBLAS Error: ") + ::gpu_solver::cublasGetErrorEnum(status) + " at " +      \
                               __FILE__ + ":" + std::to_string(__LINE__));                                             \
    }                                                                                                                  \
  } while (0)

} // namespace gpu_solver

