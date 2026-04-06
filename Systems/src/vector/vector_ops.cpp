// Compile-time backend selection (priority: BLAS > SLEEF > Highway > Sequential).
// Pass -DUSE_BLAS for OpenBLAS, -DUSE_SLEEF for SLEEF kernels, -DUSE_HWY for Highway SIMD, or neither for scalar fallback.

#if defined(USE_BLAS)
#include "impl/vector_blas.cpp"
#elif defined(USE_SLEEF)
#include "impl/vector_sleef.cpp"
#elif defined(USE_HWY) || defined(USE_HIGHWAY)
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

void matmul(const float* a, const float* b, float* out, size_t rows_a, size_t cols_a, size_t cols_b) {
    impl_matmul(a, b, out, rows_a, cols_a, cols_b);
}

void matvec(const float* a, const float* x, float* out, size_t rows, size_t cols) {
    impl_matvec(a, x, out, rows, cols);
}

void relu(const float* in, float* out, size_t n) {
    impl_relu(in, out, n);
}

void exp(const float* in, float* out, size_t n) {
    impl_exp(in, out, n);
}

void log(const float* in, float* out, size_t n) {
    impl_log(in, out, n);
}

void tanh(const float* in, float* out, size_t n) {
    impl_tanh(in, out, n);
}

void sigmoid(const float* in, float* out, size_t n) {
    impl_sigmoid(in, out, n);
}

}
