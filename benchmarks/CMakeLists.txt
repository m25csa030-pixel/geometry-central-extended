cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
  if(EXISTS "/usr/local/cuda-12.6/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.6/bin/nvcc")
  elseif(EXISTS "/usr/local/cuda/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
  endif()
endif()

project(geometry_central_benchmarks LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

find_package(CUDAToolkit REQUIRED)

find_package(Eigen3 3.3 QUIET)
if(NOT EIGEN3_FOUND AND EXISTS "/usr/include/eigen3")
  set(EIGEN3_INCLUDE_DIR "/usr/include/eigen3")
  set(EIGEN3_FOUND TRUE)
endif()

set(GC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

set(BENCH_INCLUDES
  ${GC_ROOT}/include
  ${GC_ROOT}/deps/happly
  ${GC_ROOT}/gpu_solver/include
  ${EIGEN3_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
)

add_executable(benchmark_solvers
  src/benchmark_solvers.cpp
)

target_include_directories(benchmark_solvers PRIVATE
  ${BENCH_INCLUDES}
)

target_link_libraries(benchmark_solvers PRIVATE
  ${GC_ROOT}/build/src/libgeometry-central.a
  ${GC_ROOT}/gpu_solver/build/libgpu_solver.a
  CUDA::cusparse
  CUDA::cublas
  CUDA::cudart
  pthread
)

add_executable(test_heat_method_gpu
  src/test_heat_method_gpu.cpp
)

target_include_directories(test_heat_method_gpu PRIVATE
  ${BENCH_INCLUDES}
)

target_link_libraries(test_heat_method_gpu PRIVATE
  ${GC_ROOT}/build/src/libgeometry-central.a
  ${GC_ROOT}/gpu_solver/build/libgpu_solver.a
  CUDA::cusparse
  CUDA::cublas
  CUDA::cudart
  pthread
)

add_executable(test_vector_heat_gpu
  src/test_vector_heat_gpu.cpp
)

target_include_directories(test_vector_heat_gpu PRIVATE
  ${BENCH_INCLUDES}
)

target_link_libraries(test_vector_heat_gpu PRIVATE
  ${GC_ROOT}/build/src/libgeometry-central.a
  ${GC_ROOT}/gpu_solver/build/libgpu_solver.a
  CUDA::cusparse
  CUDA::cublas
  CUDA::cudart
  pthread
)

add_executable(test_cad_robustness
  ${GC_ROOT}/tests/robustness/test_cad_robustness.cpp
)

target_include_directories(test_cad_robustness PRIVATE
  ${BENCH_INCLUDES}
)

target_link_libraries(test_cad_robustness PRIVATE
  ${GC_ROOT}/build/src/libgeometry-central.a
  ${GC_ROOT}/gpu_solver/build/libgpu_solver.a
  CUDA::cusparse
  CUDA::cublas
  CUDA::cudart
  pthread
)



