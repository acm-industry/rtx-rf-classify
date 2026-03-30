#include <cblas.h>
#include <cstddef>
#include <cmath>
#include <algorithm>

static void impl_add(const float* a, const float* b, float* out, size_t n) {
    // BLAS saxpy is in-place (y += alpha*x), doesn't match out-of-place API.
    // Use sequential fallback for add.
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

static void impl_dot(const float* a, const float* b, float* out, size_t n) {
    *out = cblas_sdot(static_cast<int>(n), a, 1, b, 1);
}

static void impl_matmul(const float* a, const float* b, float* out,
                         size_t rows_a, size_t cols_a, size_t cols_b) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                static_cast<int>(rows_a),
                static_cast<int>(cols_b),
                static_cast<int>(cols_a),
                1.0f,
                a, static_cast<int>(cols_a),
                b, static_cast<int>(cols_b),
                0.0f,
                out, static_cast<int>(cols_b));
}

static void impl_matvec(const float* a, const float* x, float* out,
                         size_t rows, size_t cols) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                static_cast<int>(rows),
                static_cast<int>(cols),
                1.0f,
                a, static_cast<int>(cols),
                x, 1,
                0.0f,
                out, 1);
}

static void impl_relu(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = std::max(0.0f, in[i]);
    }
}

static void impl_exp(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = std::exp(in[i]);
    }
}

static void impl_log(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = std::log(in[i]);
    }
}

static void impl_tanh(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = std::tanh(in[i]);
    }
}

static void impl_sigmoid(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = 1.0f / (1.0f + std::exp(-in[i]));
    }
}
