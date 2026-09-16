# Linear Solver Benchmark Results: CPU vs CUDA GPU
# Empirical Linear Solver Benchmark: CPU vs CUDA GPU in geometry-central

### System Configuration
- **GPU**: NVIDIA L4 (22563 MB VRAM, Compute 8.9)
- **CPU**: AMD EPYC 7742 64-Core Processor (256 threads)
- **CUDA Version**: 12.6 | **Driver**: 580.173.02
- **Eigen Version**: 3.4.0
- **Warmup Solves**: 3 | **Timed Runs per Entry**: 10 (mean +/- std dev)
This report documents reproducible benchmark measurements comparing CPU solvers against the CUDA Jacobi-preconditioned Conjugate Gradient (PCG) GPU solver across local mesh datasets and varying mesh resolutions in `geometry-central`.

### Benchmark Data Table
---

| Mesh | Vertices | NNZ | Operator | Solver | Prec | Iters | Rel Res | Max Error vs CPU | Setup (ms) | Solve-only (ms) | Transfer (ms) | Total (ms) |
## 1. Hardware & Software Environment

- **GPU**: NVIDIA L4
  - Architecture: Ada Lovelace (Compute Capability 8.9)
  - Memory: 22,563 MiB GDDR6 (ECC enabled)
  - Driver: 580.173.02
  - CUDA Runtime: 12.6
  - CUDA Toolchain: NVCC 12.6.85 (`/usr/local/cuda-12.6/bin/nvcc`)
- **CPU**: AMD EPYC 7742 64-Core Processor
  - Sockets: 2 | Physical Cores: 128 | Logical Threads: 256
  - Base Clock: 2.25 GHz | Boost: 3.40 GHz
- **Operating System**: Linux 5.15.0-187-generic (Ubuntu 22.04 LTS x86_64)
- **Host Compiler**: GCC 11.4.0 (Optimization `-O3 -march=native`)
- **Linear Algebra**: Eigen 3.4.0, cuSPARSE (CUDA 12.6), cuBLAS (CUDA 12.6)
- **Methodology**:
  - Warmup solves: 3 untimed executions prior to measurement to ramp GPU clocks and warm caches.
  - Timed repetitions: 10 runs per entry; values reported as $\text{mean} \pm \text{std dev}$.
  - Timing instrumentation: CUDA events (`cudaEventRecord`, `cudaEventElapsedTime`) for GPU solve, transfer, and total times; `std::chrono::high_resolution_clock` for host timings.

---

## 2. Benchmark Results Table

| Mesh | Vertices ($N$) | Nonzeros ($NNZ$) | Operator | Solver | Precision | Iters | Rel. Residual | Max Error vs CPU | Setup Time (ms) | Solve-only Time (ms) | Transfer Time (ms) | Total Time (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| bob_small | 250 | 1750 | Heat Diffusion (M + t*L) | CPU Direct (SimplicialLDLT) | FP64 | 1 | 4.14e-16 | 0.00e+00 | 2.65 | 0.36 +/- 0.00 | 0.00 +/- 0.00 | 0.36 +/- 0.00 |
| bob_small | 250 | 1750 | Heat Diffusion (M + t*L) | CPU Iterative (Eigen Jacobi-CG) | FP64 | 28 | 6.62e-07 | 1.65e-05 | 0.04 | 2.66 +/- 0.01 | 0.00 +/- 0.00 | 2.66 +/- 0.01 |
| bob_small | 250 | 1750 | Heat Diffusion (M + t*L) | GPU CUDA Jacobi-PCG | FP64 | 29 | 6.62e-07 | 1.65e-05 | 159.48 | 3.28 +/- 0.02 | 0.02 +/- 0.00 | 3.29 +/- 0.02 |
| bob_small | 250 | 1750 | Heat Diffusion (M + t*L) | GPU CUDA Jacobi-PCG | FP32 | 24 | 8.95e-06 | 2.30e-04 | 1.70 | 1.87 +/- 0.02 | 0.02 +/- 0.00 | 1.89 +/- 0.02 |
| bob_small | 250 | 1750 | Shifted Poisson (L + 1e-6*I) | CPU Direct (SimplicialLDLT) | FP64 | 1 | 3.76e-11 | 0.00e+00 | 2.60 | 0.37 +/- 0.02 | 0.00 +/- 0.00 | 0.37 +/- 0.02 |
| bob_small | 250 | 1750 | Shifted Poisson (L + 1e-6*I) | CPU Iterative (Eigen Jacobi-CG) | FP64 | 92 | 7.96e-07 | 1.44e-06 | 0.04 | 8.20 +/- 0.11 | 0.00 +/- 0.00 | 8.20 +/- 0.11 |
| bob_small | 250 | 1750 | Shifted Poisson (L + 1e-6*I) | GPU CUDA Jacobi-PCG | FP64 | 93 | 7.97e-07 | 1.47e-06 | 1.45 | 10.19 +/- 0.06 | 0.02 +/- 0.00 | 10.20 +/- 0.06 |
| bob_small | 250 | 1750 | Shifted Poisson (L + 1e-6*I) | GPU CUDA Jacobi-PCG | FP32 | 2000 | 3.89e-02 | 1.63e+02 | 1.41 | 154.07 +/- 1.24 | 0.02 +/- 0.01 | 154.09 +/- 1.24 |
| spot (base) | 2930 | 20498 | Heat Diffusion (M + t*L) | CPU Direct (SimplicialLDLT) | FP64 | 1 | 6.33e-16 | 0.00e+00 | 48.81 | 5.32 +/- 0.01 | 0.00 +/- 0.00 | 5.32 +/- 0.01 |
| spot (base) | 2930 | 20498 | Heat Diffusion (M + t*L) | CPU Iterative (Eigen Jacobi-CG) | FP64 | 49 | 8.20e-07 | 1.36e-03 | 0.43 | 51.99 +/- 0.25 | 0.00 +/- 0.00 | 51.99 +/- 0.25 |
| spot (base) | 2930 | 20498 | Heat Diffusion (M + t*L) | GPU CUDA Jacobi-PCG | FP64 | 50 | 8.20e-07 | 1.36e-03 | 1.68 | 6.05 +/- 0.01 | 0.02 +/- 0.00 | 6.07 +/- 0.01 |
| spot (base) | 2930 | 20498 | Heat Diffusion (M + t*L) | GPU CUDA Jacobi-PCG | FP32 | 42 | 8.40e-06 | 1.73e-02 | 1.54 | 3.65 +/- 0.06 | 0.02 +/- 0.00 | 3.67 +/- 0.06 |
| spot (base) | 2930 | 20498 | Shifted Poisson (L + 1e-6*I) | CPU Direct (SimplicialLDLT) | FP64 | 1 | 1.25e-11 | 0.00e+00 | 49.41 | 5.36 +/- 0.03 | 0.00 +/- 0.00 | 5.36 +/- 0.03 |
| spot (base) | 2930 | 20498 | Shifted Poisson (L + 1e-6*I) | CPU Iterative (Eigen Jacobi-CG) | FP64 | 303 | 9.47e-07 | 7.61e-06 | 0.46 | 311.53 +/- 1.07 | 0.00 +/- 0.00 | 311.53 +/- 1.07 |
| spot (base) | 2930 | 20498 | Shifted Poisson (L + 1e-6*I) | GPU CUDA Jacobi-PCG | FP64 | 304 | 9.63e-07 | 7.90e-06 | 1.83 | 36.01 +/- 0.11 | 0.02 +/- 0.00 | 36.03 +/- 0.11 |
| spot (base) | 2930 | 20498 | Shifted Poisson (L + 1e-6*I) | GPU CUDA Jacobi-PCG | FP32 | 2000 | 3.41e-02 | 1.12e+01 | 1.63 | 171.17 +/- 0.44 | 0.02 +/- 0.00 | 171.19 +/- 0.45 |
| spot (subdiv 1x) | 11714 | 81986 | Heat Diffusion (M + t*L) | CPU Direct (SimplicialLDLT) | FP64 | 1 | 6.54e-16 | 0.00e+00 | 327.59 | 25.72 +/- 0.08 | 0.00 +/- 0.00 | 25.72 +/- 0.08 |
| spot (subdiv 1x) | 11714 | 81986 | Heat Diffusion (M + t*L) | CPU Iterative (Eigen Jacobi-CG) | FP64 | 63 | 8.39e-07 | 1.15e-02 | 1.67 | 262.96 +/- 1.69 | 0.00 +/- 0.00 | 262.96 +/- 1.69 |
| spot (subdiv 1x) | 11714 | 81986 | Heat Diffusion (M + t*L) | GPU CUDA Jacobi-PCG | FP64 | 64 | 8.39e-07 | 1.15e-02 | 2.44 | 8.21 +/- 0.02 | 0.04 +/- 0.00 | 8.25 +/- 0.02 |
| spot (subdiv 1x) | 11714 | 81986 | Heat Diffusion (M + t*L) | GPU CUDA Jacobi-PCG | FP32 | 52 | 9.41e-06 | 2.41e-01 | 1.83 | 4.51 +/- 0.08 | 0.03 +/- 0.00 | 4.54 +/- 0.08 |
| spot (subdiv 1x) | 11714 | 81986 | Shifted Poisson (L + 1e-6*I) | CPU Direct (SimplicialLDLT) | FP64 | 1 | 1.72e-12 | 0.00e+00 | 324.19 | 25.47 +/- 0.24 | 0.00 +/- 0.00 | 25.47 +/- 0.24 |
| spot (subdiv 1x) | 11714 | 81986 | Shifted Poisson (L + 1e-6*I) | CPU Iterative (Eigen Jacobi-CG) | FP64 | 581 | 9.95e-07 | 2.66e-05 | 1.68 | 2381.16 +/- 3.10 | 0.00 +/- 0.00 | 2381.16 +/- 3.10 |
| spot (subdiv 1x) | 11714 | 81986 | Shifted Poisson (L + 1e-6*I) | GPU CUDA Jacobi-PCG | FP64 | 581 | 9.66e-07 | 2.75e-05 | 2.65 | 73.51 +/- 0.04 | 0.04 +/- 0.00 | 73.56 +/- 0.04 |
| spot (subdiv 1x) | 11714 | 81986 | Shifted Poisson (L + 1e-6*I) | GPU CUDA Jacobi-PCG | FP32 | 2000 | 3.93e-03 | 1.52e+00 | 1.92 | 171.26 +/- 0.23 | 0.03 +/- 0.00 | 171.29 +/- 0.23 |
| spot (subdiv 2x) | 46850 | 327938 | Heat Diffusion (M + t*L) | CPU Direct (SimplicialLDLT) | FP64 | 1 | 7.68e-16 | 0.00e+00 | 2482.53 | 127.54 +/- 0.73 | 0.00 +/- 0.00 | 127.54 +/- 0.73 |
| spot (subdiv 2x) | 46850 | 327938 | Heat Diffusion (M + t*L) | CPU Iterative (Eigen Jacobi-CG) | FP64 | 73 | 9.54e-07 | 9.03e-02 | 6.70 | 1218.59 +/- 3.16 | 0.00 +/- 0.00 | 1218.59 +/- 3.16 |
| spot (subdiv 2x) | 46850 | 327938 | Heat Diffusion (M + t*L) | GPU CUDA Jacobi-PCG | FP64 | 74 | 9.54e-07 | 9.03e-02 | 3.70 | 12.60 +/- 0.13 | 0.11 +/- 0.01 | 12.71 +/- 0.13 |
| spot (subdiv 2x) | 46850 | 327938 | Heat Diffusion (M + t*L) | GPU CUDA Jacobi-PCG | FP32 | 63 | 9.15e-06 | 7.29e-01 | 3.07 | 5.97 +/- 0.10 | 0.06 +/- 0.00 | 6.04 +/- 0.10 |
| spot (subdiv 2x) | 46850 | 327938 | Shifted Poisson (L + 1e-6*I) | CPU Direct (SimplicialLDLT) | FP64 | 1 | 1.94e-12 | 0.00e+00 | 2489.02 | 128.06 +/- 0.88 | 0.00 +/- 0.00 | 128.06 +/- 0.88 |
| spot (subdiv 2x) | 46850 | 327938 | Shifted Poisson (L + 1e-6*I) | CPU Iterative (Eigen Jacobi-CG) | FP64 | 1145 | 9.65e-07 | 6.18e-05 | 6.67 | 18715.11 +/- 51.96 | 0.00 +/- 0.00 | 18715.11 +/- 51.96 |
| spot (subdiv 2x) | 46850 | 327938 | Shifted Poisson (L + 1e-6*I) | GPU CUDA Jacobi-PCG | FP64 | 1149 | 9.72e-07 | 6.03e-05 | 3.71 | 192.63 +/- 0.41 | 0.10 +/- 0.00 | 192.74 +/- 0.41 |
| spot (subdiv 2x) | 46850 | 327938 | Shifted Poisson (L + 1e-6*I) | GPU CUDA Jacobi-PCG | FP32 | 2000 | 3.84e-02 | 1.42e-01 | 3.14 | 185.52 +/- 0.73 | 0.08 +/- 0.06 | 185.60 +/- 0.74 |
| spot (subdiv 3x) | 187394 | 1311746 | Heat Diffusion (M + t*L) | CPU Direct (SimplicialLDLT) | FP64 | 1 | 8.60e-16 | 0.00e+00 | 23178.35 | 628.90 +/- 1.54 | 0.00 +/- 0.00 | 628.90 +/- 1.54 |
| spot (subdiv 3x) | 187394 | 1311746 | Heat Diffusion (M + t*L) | CPU Iterative (Eigen Jacobi-CG) | FP64 | 90 | 9.45e-07 | 8.63e-01 | 26.84 | 5972.50 +/- 4.77 | 0.00 +/- 0.00 | 5972.50 +/- 4.77 |
| spot (subdiv 3x) | 187394 | 1311746 | Heat Diffusion (M + t*L) | GPU CUDA Jacobi-PCG | FP64 | 91 | 9.45e-07 | 8.63e-01 | 8.85 | 20.58 +/- 0.05 | 0.35 +/- 0.01 | 20.93 +/- 0.05 |
| spot (subdiv 3x) | 187394 | 1311746 | Heat Diffusion (M + t*L) | GPU CUDA Jacobi-PCG | FP32 | 74 | 9.62e-06 | 1.09e+01 | 7.93 | 8.14 +/- 0.03 | 0.19 +/- 0.00 | 8.33 +/- 0.03 |
| spot (subdiv 3x) | 187394 | 1311746 | Shifted Poisson (L + 1e-6*I) | CPU Direct (SimplicialLDLT) | FP64 | 1 | 4.61e-12 | 0.00e+00 | 23210.22 | 630.26 +/- 1.93 | 0.00 +/- 0.00 | 630.26 +/- 1.93 |
| spot (subdiv 3x) | 187394 | 1311746 | Shifted Poisson (L + 1e-6*I) | CPU Iterative (Eigen Jacobi-CG) | FP64 | 2000 | 1.35e-05 | 4.83e-03 | 26.93 | 131096.95 +/- 445.85 | 0.00 +/- 0.00 | 131096.95 +/- 445.85 |
| spot (subdiv 3x) | 187394 | 1311746 | Shifted Poisson (L + 1e-6*I) | GPU CUDA Jacobi-PCG | FP64 | 2000 | 1.48e-05 | 4.87e-03 | 9.11 | 471.22 +/- 0.28 | 0.37 +/- 0.00 | 471.59 +/- 0.28 |
| spot (subdiv 3x) | 187394 | 1311746 | Shifted Poisson (L + 1e-6*I) | GPU CUDA Jacobi-PCG | FP32 | 2000 | 1.52e-02 | 1.04e+00 | 8.26 | 258.68 +/- 0.20 | 0.20 +/- 0.00 | 258.89 +/- 0.20 |
| fox | 313 | 2179 | Heat Diffusion (M + t*L) | CPU Direct (SimplicialLDLT) | FP64 | 1 | 6.38e-16 | 0.00e+00 | 2.67 | 0.41 +/- 0.02 | 0.00 +/- 0.00 | 0.41 +/- 0.02 |
| fox | 313 | 2179 | Heat Diffusion (M + t*L) | CPU Iterative (Eigen Jacobi-CG) | FP64 | 45 | 8.11e-07 | 1.69e-08 | 0.05 | 5.29 +/- 0.03 | 0.00 +/- 0.00 | 5.29 +/- 0.03 |
| fox | 313 | 2179 | Heat Diffusion (M + t*L) | GPU CUDA Jacobi-PCG | FP64 | 46 | 8.11e-07 | 1.69e-08 | 1.69 | 5.16 +/- 0.01 | 0.02 +/- 0.00 | 5.17 +/- 0.01 |
| fox | 313 | 2179 | Heat Diffusion (M + t*L) | GPU CUDA Jacobi-PCG | FP32 | 37 | 8.28e-06 | 4.72e-07 | 1.43 | 2.89 +/- 0.01 | 0.02 +/- 0.00 | 2.90 +/- 0.01 |
| fox | 313 | 2179 | Shifted Poisson (L + 1e-6*I) | CPU Direct (SimplicialLDLT) | FP64 | 1 | 1.94e-11 | 0.00e+00 | 2.67 | 0.40 +/- 0.00 | 0.00 +/- 0.00 | 0.40 +/- 0.00 |
| fox | 313 | 2179 | Shifted Poisson (L + 1e-6*I) | CPU Iterative (Eigen Jacobi-CG) | FP64 | 156 | 8.63e-07 | 8.21e-07 | 0.05 | 17.88 +/- 0.11 | 0.00 +/- 0.00 | 17.88 +/- 0.11 |
| fox | 313 | 2179 | Shifted Poisson (L + 1e-6*I) | GPU CUDA Jacobi-PCG | FP64 | 157 | 8.66e-07 | 7.73e-07 | 1.44 | 17.44 +/- 0.06 | 0.02 +/- 0.00 | 17.45 +/- 0.06 |
| fox | 313 | 2179 | Shifted Poisson (L + 1e-6*I) | GPU CUDA Jacobi-PCG | FP32 | 2000 | 2.75e-02 | 1.49e+02 | 1.51 | 154.35 +/- 0.63 | 0.02 +/- 0.00 | 154.37 +/- 0.63 |
| cat_head | 131 | 887 | Heat Diffusion (M + t*L) | CPU Direct (SimplicialLDLT) | FP64 | 1 | 3.61e-16 | 0.00e+00 | 1.11 | 0.17 +/- 0.00 | 0.00 +/- 0.00 | 0.17 +/- 0.00 |
| cat_head | 131 | 887 | Heat Diffusion (M + t*L) | CPU Iterative (Eigen Jacobi-CG) | FP64 | 25 | 4.77e-07 | 9.66e-06 | 0.02 | 1.31 +/- 0.02 | 0.00 +/- 0.00 | 1.31 +/- 0.02 |
| cat_head | 131 | 887 | Heat Diffusion (M + t*L) | GPU CUDA Jacobi-PCG | FP64 | 26 | 4.77e-07 | 9.66e-06 | 1.49 | 2.76 +/- 0.02 | 0.02 +/- 0.00 | 2.77 +/- 0.02 |
| cat_head | 131 | 887 | Heat Diffusion (M + t*L) | GPU CUDA Jacobi-PCG | FP32 | 22 | 6.65e-06 | 6.40e-05 | 1.42 | 1.75 +/- 0.01 | 0.02 +/- 0.00 | 1.76 +/- 0.01 |
| cat_head | 131 | 887 | Shifted Poisson (L + 1e-6*I) | CPU Direct (SimplicialLDLT) | FP64 | 1 | 3.86e-11 | 0.00e+00 | 1.11 | 0.17 +/- 0.00 | 0.00 +/- 0.00 | 0.17 +/- 0.00 |
| cat_head | 131 | 887 | Shifted Poisson (L + 1e-6*I) | CPU Iterative (Eigen Jacobi-CG) | FP64 | 68 | 7.75e-07 | 9.57e-07 | 0.02 | 3.36 +/- 0.01 | 0.00 +/- 0.00 | 3.36 +/- 0.01 |
| cat_head | 131 | 887 | Shifted Poisson (L + 1e-6*I) | GPU CUDA Jacobi-PCG | FP64 | 69 | 7.75e-07 | 1.28e-06 | 1.42 | 7.10 +/- 0.02 | 0.02 +/- 0.00 | 7.12 +/- 0.02 |
| cat_head | 131 | 887 | Shifted Poisson (L + 1e-6*I) | GPU CUDA Jacobi-PCG | FP32 | 2000 | 1.39e-01 | 8.61e+02 | 1.52 | 154.71 +/- 0.38 | 0.02 +/- 0.00 | 154.73 +/- 0.38 |
| **bob_small** | 250 | 1,750 | Heat ($M + tL$) | CPU Direct (SimplicialLDLT) | FP64 | 1 | $4.14 \times 10^{-16}$ | 0.00 | 2.65 | $0.36 \pm 0.00$ | 0.00 | **$0.36 \pm 0.00$** |
| bob_small | 250 | 1,750 | Heat ($M + tL$) | CPU Iterative (Eigen CG) | FP64 | 28 | $6.62 \times 10^{-7}$ | $1.65 \times 10^{-5}$ | 0.04 | $2.66 \pm 0.01$ | 0.00 | $2.66 \pm 0.01$ |
| bob_small | 250 | 1,750 | Heat ($M + tL$) | GPU CUDA-PCG | FP64 | 29 | $6.62 \times 10^{-7}$ | $1.65 \times 10^{-5}$ | 159.48 | $3.28 \pm 0.02$ | $0.02 \pm 0.00$ | $3.29 \pm 0.02$ |
| bob_small | 250 | 1,750 | Heat ($M + tL$) | GPU CUDA-PCG | FP32 | 24 | $8.95 \times 10^{-6}$ | $2.30 \times 10^{-4}$ | 1.70 | $1.87 \pm 0.02$ | $0.02 \pm 0.00$ | $1.89 \pm 0.02$ |
| **bob_small** | 250 | 1,750 | Poisson ($L + 10^{-6}I$) | CPU Direct (SimplicialLDLT) | FP64 | 1 | $3.76 \times 10^{-11}$ | 0.00 | 2.60 | $0.37 \pm 0.02$ | 0.00 | **$0.37 \pm 0.02$** |
| bob_small | 250 | 1,750 | Poisson ($L + 10^{-6}I$) | CPU Iterative (Eigen CG) | FP64 | 92 | $7.96 \times 10^{-7}$ | $1.44 \times 10^{-6}$ | 0.04 | $8.20 \pm 0.11$ | 0.00 | $8.20 \pm 0.11$ |
| bob_small | 250 | 1,750 | Poisson ($L + 10^{-6}I$) | GPU CUDA-PCG | FP64 | 93 | $7.97 \times 10^{-7}$ | $1.47 \times 10^{-6}$ | 1.45 | $10.19 \pm 0.06$ | $0.02 \pm 0.00$ | $10.20 \pm 0.06$ |
| bob_small | 250 | 1,750 | Poisson ($L + 10^{-6}I$) | GPU CUDA-PCG | FP32 | 2000 | $3.89 \times 10^{-2}$ | $1.63 \times 10^{2}$ | 1.41 | $154.07 \pm 1.24$ | $0.02 \pm 0.01$ | $154.09 \pm 1.24$ |
| **spot (base)** | 2,930 | 20,498 | Heat ($M + tL$) | CPU Direct (SimplicialLDLT) | FP64 | 1 | $6.33 \times 10^{-16}$ | 0.00 | 48.81 | $5.32 \pm 0.01$ | 0.00 | $5.32 \pm 0.01$ |
| spot (base) | 2,930 | 20,498 | Heat ($M + tL$) | CPU Iterative (Eigen CG) | FP64 | 49 | $8.20 \times 10^{-7}$ | $1.36 \times 10^{-3}$ | 0.43 | $51.99 \pm 0.25$ | 0.00 | $51.99 \pm 0.25$ |
| spot (base) | 2,930 | 20,498 | Heat ($M + tL$) | GPU CUDA-PCG | FP64 | 50 | $8.20 \times 10^{-7}$ | $1.36 \times 10^{-3}$ | 1.68 | $6.05 \pm 0.01$ | $0.02 \pm 0.00$ | $6.07 \pm 0.01$ |
| spot (base) | 2,930 | 20,498 | Heat ($M + tL$) | GPU CUDA-PCG | FP32 | 42 | $8.40 \times 10^{-6}$ | $1.73 \times 10^{-2}$ | 1.54 | $3.65 \pm 0.06$ | $0.02 \pm 0.00$ | **$3.67 \pm 0.06$** |
| **spot (base)** | 2,930 | 20,498 | Poisson ($L + 10^{-6}I$) | CPU Direct (SimplicialLDLT) | FP64 | 1 | $1.25 \times 10^{-11}$ | 0.00 | 49.41 | $5.36 \pm 0.03$ | 0.00 | **$5.36 \pm 0.03$** |
| spot (base) | 2,930 | 20,498 | Poisson ($L + 10^{-6}I$) | CPU Iterative (Eigen CG) | FP64 | 303 | $9.47 \times 10^{-7}$ | $7.61 \times 10^{-6}$ | 0.46 | $311.53 \pm 1.07$ | 0.00 | $311.53 \pm 1.07$ |
| spot (base) | 2,930 | 20,498 | Poisson ($L + 10^{-6}I$) | GPU CUDA-PCG | FP64 | 304 | $9.63 \times 10^{-7}$ | $7.90 \times 10^{-6}$ | 1.83 | $36.01 \pm 0.11$ | $0.02 \pm 0.00$ | $36.03 \pm 0.11$ |
| spot (base) | 2,930 | 20,498 | Poisson ($L + 10^{-6}I$) | GPU CUDA-PCG | FP32 | 2000 | $3.41 \times 10^{-2}$ | $1.12 \times 10^{1}$ | 1.63 | $171.17 \pm 0.44$ | $0.02 \pm 0.00$ | $171.19 \pm 0.45$ |
| **spot (subdiv 1x)** | 11,714 | 81,986 | Heat ($M + tL$) | CPU Direct (SimplicialLDLT) | FP64 | 1 | $6.54 \times 10^{-16}$ | 0.00 | 327.59 | $25.72 \pm 0.08$ | 0.00 | $25.72 \pm 0.08$ |
| spot (subdiv 1x) | 11,714 | 81,986 | Heat ($M + tL$) | CPU Iterative (Eigen CG) | FP64 | 63 | $8.39 \times 10^{-7}$ | $1.15 \times 10^{-2}$ | 1.67 | $262.96 \pm 1.69$ | 0.00 | $262.96 \pm 1.69$ |
| spot (subdiv 1x) | 11,714 | 81,986 | Heat ($M + tL$) | GPU CUDA-PCG | FP64 | 64 | $8.39 \times 10^{-7}$ | $1.15 \times 10^{-2}$ | 2.44 | $8.21 \pm 0.02$ | $0.04 \pm 0.00$ | **$8.25 \pm 0.02$** |
| spot (subdiv 1x) | 11,714 | 81,986 | Heat ($M + tL$) | GPU CUDA-PCG | FP32 | 52 | $9.41 \times 10^{-6}$ | $2.41 \times 10^{-1}$ | 1.83 | $4.51 \pm 0.08$ | $0.03 \pm 0.00$ | **$4.54 \pm 0.08$** |
| **spot (subdiv 1x)** | 11,714 | 81,986 | Poisson ($L + 10^{-6}I$) | CPU Direct (SimplicialLDLT) | FP64 | 1 | $1.72 \times 10^{-12}$ | 0.00 | 324.19 | $25.47 \pm 0.24$ | 0.00 | **$25.47 \pm 0.24$** |
| spot (subdiv 1x) | 11,714 | 81,986 | Poisson ($L + 10^{-6}I$) | CPU Iterative (Eigen CG) | FP64 | 581 | $9.95 \times 10^{-7}$ | $2.66 \times 10^{-5}$ | 1.68 | $2381.16 \pm 3.10$ | 0.00 | $2381.16 \pm 3.10$ |
| spot (subdiv 1x) | 11,714 | 81,986 | Poisson ($L + 10^{-6}I$) | GPU CUDA-PCG | FP64 | 581 | $9.66 \times 10^{-7}$ | $2.75 \times 10^{-5}$ | 2.65 | $73.51 \pm 0.04$ | $0.04 \pm 0.00$ | $73.56 \pm 0.04$ |
| spot (subdiv 1x) | 11,714 | 81,986 | Poisson ($L + 10^{-6}I$) | GPU CUDA-PCG | FP32 | 2000 | $3.93 \times 10^{-3}$ | $1.52 \times 10^{0}$ | 1.92 | $171.26 \pm 0.23$ | $0.03 \pm 0.00$ | $171.29 \pm 0.23$ |
| **spot (subdiv 2x)** | 46,850 | 327,938 | Heat ($M + tL$) | CPU Direct (SimplicialLDLT) | FP64 | 1 | $7.68 \times 10^{-16}$ | 0.00 | 2482.53 | $127.54 \pm 0.73$ | 0.00 | $127.54 \pm 0.73$ |
| spot (subdiv 2x) | 46,850 | 327,938 | Heat ($M + tL$) | CPU Iterative (Eigen CG) | FP64 | 73 | $9.54 \times 10^{-7}$ | $9.03 \times 10^{-2}$ | 6.70 | $1218.59 \pm 3.16$ | 0.00 | $1218.59 \pm 3.16$ |
| spot (subdiv 2x) | 46,850 | 327,938 | Heat ($M + tL$) | GPU CUDA-PCG | FP64 | 74 | $9.54 \times 10^{-7}$ | $9.03 \times 10^{-2}$ | 3.70 | $12.60 \pm 0.13$ | $0.11 \pm 0.01$ | **$12.71 \pm 0.13$** |
| spot (subdiv 2x) | 46,850 | 327,938 | Heat ($M + tL$) | GPU CUDA-PCG | FP32 | 63 | $9.15 \times 10^{-6}$ | $7.29 \times 10^{-1}$ | 3.07 | $5.97 \pm 0.10$ | $0.06 \pm 0.00$ | **$6.04 \pm 0.10$** |
| **spot (subdiv 2x)** | 46,850 | 327,938 | Poisson ($L + 10^{-6}I$) | CPU Direct (SimplicialLDLT) | FP64 | 1 | $1.94 \times 10^{-12}$ | 0.00 | 2489.02 | $128.06 \pm 0.88$ | 0.00 | **$128.06 \pm 0.88$** |
| spot (subdiv 2x) | 46,850 | 327,938 | Poisson ($L + 10^{-6}I$) | CPU Iterative (Eigen CG) | FP64 | 1145 | $9.65 \times 10^{-7}$ | $6.18 \times 10^{-5}$ | 6.67 | $18715.11 \pm 51.96$ | 0.00 | $18715.11 \pm 51.96$ |
| spot (subdiv 2x) | 46,850 | 327,938 | Poisson ($L + 10^{-6}I$) | GPU CUDA-PCG | FP64 | 1149 | $9.72 \times 10^{-7}$ | $6.03 \times 10^{-5}$ | 3.71 | $192.63 \pm 0.41$ | $0.10 \pm 0.00$ | $192.74 \pm 0.41$ |
| spot (subdiv 2x) | 46,850 | 327,938 | Poisson ($L + 10^{-6}I$) | GPU CUDA-PCG | FP32 | 2000 | $3.84 \times 10^{-2}$ | $1.42 \times 10^{-1}$ | 3.14 | $185.52 \pm 0.73$ | $0.08 \pm 0.06$ | $185.60 \pm 0.74$ |
| **spot (subdiv 3x)** | 187,394 | 1,311,746 | Heat ($M + tL$) | CPU Direct (SimplicialLDLT) | FP64 | 1 | $8.60 \times 10^{-16}$ | 0.00 | 23178.35 | $628.90 \pm 1.54$ | 0.00 | $628.90 \pm 1.54$ |
| spot (subdiv 3x) | 187,394 | 1,311,746 | Heat ($M + tL$) | CPU Iterative (Eigen CG) | FP64 | 90 | $9.45 \times 10^{-7}$ | $8.63 \times 10^{-1}$ | 26.84 | $5972.50 \pm 4.77$ | 0.00 | $5972.50 \pm 4.77$ |
| spot (subdiv 3x) | 187,394 | 1,311,746 | Heat ($M + tL$) | GPU CUDA-PCG | FP64 | 91 | $9.45 \times 10^{-7}$ | $8.63 \times 10^{-1}$ | 8.85 | $20.58 \pm 0.05$ | $0.35 \pm 0.01$ | **$20.93 \pm 0.05$** |
| spot (subdiv 3x) | 187,394 | 1,311,746 | Heat ($M + tL$) | GPU CUDA-PCG | FP32 | 74 | $9.62 \times 10^{-6}$ | $1.09 \times 10^{1}$ | 7.93 | $8.14 \pm 0.03$ | $0.19 \pm 0.00$ | **$8.33 \pm 0.03$** |
| **spot (subdiv 3x)** | 187,394 | 1,311,746 | Poisson ($L + 10^{-6}I$) | CPU Direct (SimplicialLDLT) | FP64 | 1 | $4.61 \times 10^{-12}$ | 0.00 | 23210.22 | $630.26 \pm 1.93$ | 0.00 | **$630.26 \pm 1.93$** |
| spot (subdiv 3x) | 187,394 | 1,311,746 | Poisson ($L + 10^{-6}I$) | CPU Iterative (Eigen CG) | FP64 | 2000 | $1.35 \times 10^{-5}$ | $4.83 \times 10^{-3}$ | 26.93 | $131096.95 \pm 445.85$ | 0.00 | $131096.95 \pm 445.85$ |
| spot (subdiv 3x) | 187,394 | 1,311,746 | Poisson ($L + 10^{-6}I$) | GPU CUDA-PCG | FP64 | 2000 | $1.48 \times 10^{-5}$ | $4.87 \times 10^{-3}$ | 9.11 | $471.22 \pm 0.28$ | $0.37 \pm 0.00$ | **$471.59 \pm 0.28$** |
| spot (subdiv 3x) | 187,394 | 1,311,746 | Poisson ($L + 10^{-6}I$) | GPU CUDA-PCG | FP32 | 2000 | $1.52 \times 10^{-2}$ | $1.04 \times 10^{0}$ | 8.26 | $258.68 \pm 0.20$ | $0.20 \pm 0.00$ | $258.89 \pm 0.20$ |
| **fox** | 313 | 2,179 | Heat ($M + tL$) | CPU Direct (SimplicialLDLT) | FP64 | 1 | $6.38 \times 10^{-16}$ | 0.00 | 2.67 | $0.41 \pm 0.02$ | 0.00 | **$0.41 \pm 0.02$** |
| fox | 313 | 2,179 | Heat ($M + tL$) | GPU CUDA-PCG | FP64 | 46 | $8.11 \times 10^{-7}$ | $1.69 \times 10^{-8}$ | 1.69 | $5.16 \pm 0.01$ | $0.02 \pm 0.00$ | $5.17 \pm 0.01$ |
| **cat_head** | 131 | 887 | Heat ($M + tL$) | CPU Direct (SimplicialLDLT) | FP64 | 1 | $3.61 \times 10^{-16}$ | 0.00 | 1.11 | $0.17 \pm 0.00$ | 0.00 | **$0.17 \pm 0.00$** |
| cat_head | 131 | 887 | Heat ($M + tL$) | GPU CUDA-PCG | FP64 | 26 | $4.77 \times 10^{-7}$ | $9.66 \times 10^{-6}$ | 1.49 | $2.76 \pm 0.02$ | $0.02 \pm 0.00$ | $2.77 \pm 0.02$ |

---

## 3. Key Findings & Performance Analysis

### 3.1. Scaling with Mesh Size & The Crossover Point
1. **Small Meshes ($N < 3,000$ vertices)**:
   - For tiny meshes ($N = 131$ to $N = 313$), the CPU direct solver (`SimplicialLDLT`) takes only $\approx 0.17 - 0.41$ ms.
   - GPU PCG incurs kernel launch overhead and stream synchronization ($\approx 2.5 - 5$ ms).
   - **Crossover Point**: GPU PCG surpasses CPU direct solve at approximately **$N \approx 3,000$ to $5,000$ vertices** for heat diffusion.
2. **Medium Meshes ($N = 10,000 - 50,000$ vertices)**:
   - At $N = 11,714$ (`spot` 1x): GPU PCG solves heat diffusion in **$8.21$ ms** vs CPU Direct's **$25.72$ ms** (**$3.1\times$ faster**).
   - At $N = 46,850$ (`spot` 2x): GPU PCG solves heat diffusion in **$12.60$ ms** vs CPU Direct's **$127.54$ ms** (**$10.1\times$ faster**), and is **$96\times$ faster** than CPU Eigen PCG ($1,218$ ms).
3. **Large Meshes ($N \ge 187,000$ vertices, $NNZ > 1.3\text{M}$)**:
   - At $N = 187,394$ (`spot` 3x):
     - GPU PCG solve-only time is **$20.58$ ms** (total $20.93$ ms).
     - CPU direct solve is **$628.90$ ms** (**$30.1\times$ slower** than GPU).
     - CPU direct setup/factorization takes **$23,178$ ms** ($23.2$ seconds!).
     - GPU setup takes **$8.85$ ms** (**$2,600\times$ faster setup**).
     - For a one-off solve (factorization + solve): GPU is **$800\times$ faster** ($29.8$ ms total vs $23,807$ ms).

### 3.2. Solve-Only Time vs. PCIe Memory Transfer Overhead
The benchmark measured host-to-device ($b$) and device-to-host ($x$) transfer times separately via CUDA events:
- At $N = 2,930$: Transfer time is $0.02$ ms ($0.3\%$ of total $6.07$ ms).
- At $N = 11,714$: Transfer time is $0.04$ ms ($0.5\%$ of total $8.25$ ms).
- At $N = 46,850$: Transfer time is $0.11$ ms ($0.9\%$ of total $12.71$ ms).
- At $N = 187,394$: Transfer time is $0.35$ ms ($1.7\%$ of total $20.93$ ms).
- **Conclusion**: Thanks to zero runtime buffer allocations in `solve()`, **PCIe transfer overhead is negligible ($< 2\%$ across all mesh sizes)**. Device compute and memory bandwidth completely dominate the runtime.

### 3.3. Operator Conditioning: Heat Flow vs. Shifted Poisson
- **Heat Diffusion Operator ($M + tL$)**:
  - Extremely well-conditioned due to the mass matrix diagonal dominance ($M$).
  - Requires only $25 - 90$ PCG iterations regardless of mesh size.
  - GPU PCG scales sub-linearly: $20.58$ ms for $187\text{k}$ vertices.
- **Shifted Poisson Operator ($L + 10^{-6}I$)**:
  - Pure cotan Laplacian has a 1D null space. Shifting by $10^{-6}I$ results in an extreme condition number ($\kappa \sim 10^{10}$).
  - Iteration counts grow significantly with mesh resolution ($93$ iters at $N=250 \to 304$ at $N=2,930 \to 1,149$ at $N=46,850 \to 2,000$ at $N=187,394$).
  - CPU iterative CG becomes impractical on large meshes: at $N = 46,850$, CPU CG requires $18.7$ seconds; at $N = 187,394$, CPU CG takes **$131$ seconds** (over 2 minutes).
  - GPU PCG finishes the 2,000 iterations in **$471.22$ ms** (**$278\times$ faster than CPU CG**), achieving relative residual $1.48 \times 10^{-5}$.
  - However, for shifted Poisson at $N \le 46,850$, CPU direct Cholesky solve-only time ($128$ ms) is competitive with GPU PCG ($192$ ms) because PCG requires $>1,000$ iterations due to ill-conditioning.

### 3.4. Numerical Precision: FP64 (Double) vs. FP32 (Float)
- **Heat Diffusion**: FP32 converges in slightly fewer iterations than FP64 ($74$ vs $91$ iters on $N=187\text{k}$) and runs roughly **$2.5\times$ faster** ($8.14$ ms vs $20.58$ ms).
- **Shifted Poisson**: **FP32 fails to converge** (hits 2,000 max iterations with stagnant relative residual $\approx 10^{-2} - 10^{-3}$). Single-precision 24-bit mantissa cannot resolve the high condition number ($\kappa > 10^7$) of the shifted Laplacian, causing catastrophic roundoff cancellation.
- **Verdict**: As predicted in `CUDA_DESIGN.md`, **double precision (FP64) is strictly required** for differential geometric operators. FP32 can only be used safely on well-conditioned heat diffusion systems.
