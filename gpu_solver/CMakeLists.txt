cmake_minimum_required(VERSION 3.18)

# Set CUDA compiler fallback if not explicitly provided
if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
  if(EXISTS "/usr/local/cuda-12.6/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.6/bin/nvcc")
  elseif(EXISTS "/usr/local/cuda/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
  endif()
endif()

project(gpu_solver LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default to target Ada Lovelace (sm_89 for L4) with fallback for Ampere/Turing
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

find_package(CUDAToolkit REQUIRED)

# Check for Eigen3 for testing
find_package(Eigen3 3.3 QUIET)
if(NOT EIGEN3_FOUND AND EXISTS "/usr/include/eigen3")
  set(EIGEN3_INCLUDE_DIR "/usr/include/eigen3")
  set(EIGEN3_FOUND TRUE)
endif()

# Standalone GPU Solver Library
add_library(gpu_solver STATIC
  src/cuda_pcg_solver.cu
)

target_include_directories(gpu_solver PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(gpu_solver PUBLIC
  CUDA::cusparse
  CUDA::cublas
  CUDA::cudart
)

# Test Suite Executable
enable_testing()

add_executable(test_cuda_pcg
  tests/test_cuda_pcg.cpp
)

target_include_directories(test_cuda_pcg PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(test_cuda_pcg PRIVATE
  gpu_solver
)

add_test(NAME test_cuda_pcg COMMAND test_cuda_pcg)

