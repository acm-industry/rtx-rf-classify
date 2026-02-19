#ifndef VEC_OPS_H
#define VEC_OPS_H

#include <cstddef>

/*
    Purpose: Vector operations interface supporting scalar, SIMD, and BLAS backends.

    Backend selection (compile-time, highest priority first):
        1. BLAS (OpenBLAS):  -DUSE_BLAS    — delegates matmul/gemv/dot to CBLAS routines
        2. Highway SIMD:     -DUSE_HIGHWAY — portable SIMD via Google Highway
        3. Sequential:       (default)     — plain scalar loops

    Usage: Include this header and link against the corresponding implementation file.
        Scalar:   g++ -std=c++17 -o my_program my_program.cpp vector_ops.cpp
        Highway:  g++ -std=c++17 -DUSE_HIGHWAY -o my_program my_program.cpp vector_ops.cpp -I "/path/to/highway" -lhwy
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

}

#endif
