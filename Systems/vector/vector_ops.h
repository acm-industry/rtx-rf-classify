#ifndef VEC_OPS_H
#define VEC_OPS_H

#include <cstddef>

// Compile with -DUSE_HIGHWAY to use the Highway SIMD backend.
// Without it, a sequential implementation will be used.

namespace vec {

// out[n] = a[n] + b[n]
void add(const float* a, const float* b, float* out, size_t n);

// out = a[n] . b[n]
void dot(const float* a, const float* b, float* out, size_t n);

// out[rows_a x cols_b] = A[rows_a x cols_a] * B[cols_a x cols_b], row-major.
void matmul(const float* a, const float* b, float* out,
            size_t rows_a, size_t cols_a, size_t cols_b);

// out[rows] = A[rows x cols] * x[cols], row-major.
void matvec(const float* a, const float* x, float* out,
            size_t rows, size_t cols);

}

#endif
