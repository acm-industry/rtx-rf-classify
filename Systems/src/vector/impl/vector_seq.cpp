#include <cstddef>
#include <cmath>
#include <cstdlib>

static constexpr float PI = 3.14159265358979323846f;

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
