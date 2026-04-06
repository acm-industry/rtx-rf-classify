#include <cstddef>
#include <cmath>
#include <algorithm>

#if __has_include(<sleef.h>)
#include <sleef.h>
#define VEC_SLEEF_HEADER_FOUND 1
#else
#define VEC_SLEEF_HEADER_FOUND 0
#endif

// NOTE:
// - This backend is selected only when USE_SLEEF is enabled.
// - We call SLEEF math entry points for transcendental unary ops when sleef.h
//   is available, and fall back to std::* otherwise.
// - add/dot/matmul/matvec remain simple loops here; BLAS still owns matrix ops.

static inline float sleef_exp_u10(float x) {
#if VEC_SLEEF_HEADER_FOUND
    return Sleef_expf_u10(x);
#else
    return std::exp(x);
#endif
}

static inline float sleef_log_u10(float x) {
#if VEC_SLEEF_HEADER_FOUND
    return Sleef_logf_u10(x);
#else
    return std::log(x);
#endif
}

static inline float sleef_tanh_u10(float x) {
#if VEC_SLEEF_HEADER_FOUND
    return Sleef_tanhf_u10(x);
#else
    return std::tanh(x);
#endif
}

static void impl_add(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

static void impl_dot(const float* a, const float* b, float* out, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    *out = sum;
}

static void impl_matmul(const float* a, const float* b, float* out,
                        size_t rows_a, size_t cols_a, size_t cols_b) {
    for (size_t i = 0; i < rows_a; ++i) {
        for (size_t j = 0; j < cols_b; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < cols_a; ++k) {
                sum += a[i * cols_a + k] * b[k * cols_b + j];
            }
            out[i * cols_b + j] = sum;
        }
    }
}

static void impl_matvec(const float* a, const float* x, float* out,
                        size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < cols; ++j) {
            sum += a[i * cols + j] * x[j];
        }
        out[i] = sum;
    }
}

static void impl_relu(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = std::max(0.0f, in[i]);
    }
}

static void impl_exp(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = sleef_exp_u10(in[i]);
    }
}

static void impl_log(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = sleef_log_u10(in[i]);
    }
}

static void impl_tanh(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = sleef_tanh_u10(in[i]);
    }
}

static void impl_sigmoid(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = 1.0f / (1.0f + sleef_exp_u10(-in[i]));
    }
}
