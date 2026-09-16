#pragma once

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu_solver {

/**
 * @brief Compressed Sparse Row (CSR) matrix representation in host memory.
 *
 * 0-indexed CSR storage:
 * - rowOffsets: size nRows + 1
 * - colIndices: size nnz
 * - values: size nnz
 */
template <typename T>
struct CsrMatrix {
  int nRows = 0;
  int nCols = 0;
  int nnz = 0;

  std::vector<int> rowOffsets;
  std::vector<int> colIndices;
  std::vector<T> values;

  CsrMatrix() = default;

  CsrMatrix(int rows, int cols, int numNonZeros)
      : nRows(rows), nCols(cols), nnz(numNonZeros), rowOffsets(rows + 1, 0), colIndices(numNonZeros, 0), values(numNonZeros, 0) {}

  CsrMatrix(int rows, int cols, std::vector<int> offsets, std::vector<int> indices, std::vector<T> vals)
      : nRows(rows), nCols(cols), nnz(static_cast<int>(vals.size())),
        rowOffsets(std::move(offsets)), colIndices(std::move(indices)), values(std::move(vals)) {
    validate();
  }

  void validate() const {
    if (nRows < 0 || nCols < 0 || nnz < 0) {
      throw std::invalid_argument("Matrix dimensions and nnz must be non-negative.");
    }
    if (static_cast<int>(rowOffsets.size()) != nRows + 1) {
      throw std::invalid_argument("rowOffsets size must be nRows + 1.");
    }
    if (static_cast<int>(colIndices.size()) != nnz || static_cast<int>(values.size()) != nnz) {
      throw std::invalid_argument("colIndices and values sizes must match nnz.");
    }
    if (rowOffsets[0] != 0 || rowOffsets[nRows] != nnz) {
      throw std::invalid_argument("rowOffsets must start at 0 and end at nnz.");
    }
    for (int i = 0; i < nRows; ++i) {
      if (rowOffsets[i] > rowOffsets[i + 1]) {
        throw std::invalid_argument("rowOffsets must be monotonically non-decreasing.");
      }
      for (int j = rowOffsets[i]; j < rowOffsets[i + 1]; ++j) {
        if (colIndices[j] < 0 || colIndices[j] >= nCols) {
          throw std::invalid_argument("Column index out of bounds: " + std::to_string(colIndices[j]));
        }
        if (std::isnan(static_cast<double>(values[j])) || std::isinf(static_cast<double>(values[j]))) {
          throw std::invalid_argument("Matrix value is NaN or Inf at index: " + std::to_string(j));
        }
      }
    }
  }

  /**
   * @brief CPU reference matrix-vector multiplication: y = alpha * A * x + beta * y
   */
  void spmv(const T* x, T* y, T alpha = 1, T beta = 0) const {
    for (int i = 0; i < nRows; ++i) {
      T rowSum = 0;
      for (int j = rowOffsets[i]; j < rowOffsets[i + 1]; ++j) {
        rowSum += values[j] * x[colIndices[j]];
      }
      y[i] = alpha * rowSum + beta * y[i];
    }
  }

  /**
   * @brief Factory for diagonal matrix.
   */
  static CsrMatrix<T> fromDiagonal(const std::vector<T>& diag) {
    int n = static_cast<int>(diag.size());
    CsrMatrix<T> mat(n, n, n);
    for (int i = 0; i < n; ++i) {
      mat.rowOffsets[i] = i;
      mat.colIndices[i] = i;
      mat.values[i] = diag[i];
    }
    mat.rowOffsets[n] = n;
    return mat;
  }

  /**
   * @brief Factory for 1D tridiagonal Poisson/Laplacian matrix with optional diagonal regularization:
   * [-1, 2 + shift, -1].
   */
  static CsrMatrix<T> make1DLaplacian(int n, T diagShift = 0) {
    if (n <= 0) {
      throw std::invalid_argument("Size must be positive.");
    }
    int estimatedNnz = 3 * n - 2;
    CsrMatrix<T> mat;
    mat.nRows = n;
    mat.nCols = n;
    mat.rowOffsets.resize(n + 1, 0);
    mat.colIndices.reserve(estimatedNnz);
    mat.values.reserve(estimatedNnz);

    int count = 0;
    for (int i = 0; i < n; ++i) {
      mat.rowOffsets[i] = count;
      if (i > 0) {
        mat.colIndices.push_back(i - 1);
        mat.values.push_back(static_cast<T>(-1));
        count++;
      }
      mat.colIndices.push_back(i);
      mat.values.push_back(static_cast<T>(2) + diagShift);
      count++;
      if (i < n - 1) {
        mat.colIndices.push_back(i + 1);
        mat.values.push_back(static_cast<T>(-1));
        count++;
      }
    }
    mat.rowOffsets[n] = count;
    mat.nnz = count;
    mat.validate();
    return mat;
  }
};

} // namespace gpu_solver

