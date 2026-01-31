// Compile-time backend selection.
// Pass -DUSE_HIGHWAY to use the Highway SIMD backend.
// Without it, the plain scalar fallback is compiled.

#ifdef USE_HIGHWAY
#include "impl/vector_hwy.cpp"
#else
#include "impl/vector_seq.cpp"
#endif

#include "vector_ops.h"

namespace vec {

void add(const float* a, const float* b, float* out, size_t n) {
    impl_add(a, b, out, n);
}

void dot(const float* a, const float* b, float* out, size_t n) {
    impl_dot(a, b, out, n);
}

void matmul(const float* a, const float* b, float* out,
            size_t rows_a, size_t cols_a, size_t cols_b) {
    impl_matmul(a, b, out, rows_a, cols_a, cols_b);
}

void matvec(const float* a, const float* x, float* out,
            size_t rows, size_t cols) {
    impl_matvec(a, x, out, rows, cols);
}

} // namespace vec
