#ifndef VEC_OPS_H
#define VEC_OPS_H

#include <cstddef>

/*
    Purpose: Vector operations interface supporting scalar, SIMD, and BLAS backends.

    Backend selection (compile-time, highest priority first):
        1. BLAS (OpenBLAS):  -DUSE_BLAS    — delegates matmul/gemv/dot to CBLAS routines
        2. SLEEF SIMD:       -DUSE_SLEEF   — SIMD math-kernel backend
        3. Highway SIMD:     -DUSE_HWY     — portable SIMD via Google Highway
        4. Sequential:       (default)     — plain scalar loops

    Usage: Include this header and link against the corresponding implementation file.
        Scalar:   g++ -std=c++17 -o my_program my_program.cpp vector_ops.cpp
        Highway:  g++ -std=c++17 -DUSE_HWY -o my_program my_program.cpp vector_ops.cpp -I "/path/to/highway" -lhwy
        BLAS:     g++ -std=c++17 -DUSE_BLAS -o my_program my_program.cpp vector_ops.cpp -I "/path/to/openblas/include" -lopenblas

    Highway Installation: https://google.github.io/highway/en/master/README.html#installation
        Mac: brew install highway
*/

namespace vec {

// out[n] = a[n] + b[n]
void add(const float* a, const float* b, float* out, size_t n);

// out = a[n] . b[n]
void dot(const float* a, const float* b, float* out, size_t n);

// out[rows_a x cols_b] = A[rows_a x cols_a] * B[cols_a x cols_b], row-major
void matmul(const float* a, const float* b, float* out, size_t rows_a, size_t cols_a, size_t cols_b);

// out[rows] = A[rows x cols] * x[cols], row-major
void matvec(const float* a, const float* x, float* out, size_t rows, size_t cols);

// out[n] = max(0, in[n])
void relu(const float* in, float* out, size_t n);

// out[n] = exp(in[n])
void exp(const float* in, float* out, size_t n);

// out[n] = log(in[n])
void log(const float* in, float* out, size_t n);

// out[n] = tanh(in[n])
void tanh(const float* in, float* out, size_t n);

// out[n] = 1 / (1 + exp(-in[n]))
void sigmoid(const float* in, float* out, size_t n);

}

#endif
