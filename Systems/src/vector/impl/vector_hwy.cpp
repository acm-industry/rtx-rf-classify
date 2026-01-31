#include "../vector_ops.h"
#include <hwy/highway.h>
#include <cmath>
#include <cstdlib>

static constexpr float PI = 3.14159265358979323846f;

HWY_BEFORE_NAMESPACE();

namespace hwy {
namespace HWY_NAMESPACE {

// --- add ---

void hwy_add(const float* a, const float* b, float* out, size_t n) {
    const ScalableTag<float> d;
    const size_t lanes = Lanes(d);

    size_t i = 0;
    while (i < n) {
        const size_t remaining = n - i;
        const size_t count = remaining < lanes ? remaining : lanes;
        auto va = LoadN(d, a + i, count);
        auto vb = LoadN(d, b + i, count);
        StoreN(Add(va, vb), d, out + i, count);
        i += count;
    }
}


// --- dot ---

void hwy_dot(const float* a, const float* b, float* out, size_t n) {
    const ScalableTag<float> d;
    const size_t lanes = Lanes(d);
    auto sum = Zero(d);

    size_t i = 0;
    while (i < n) {
        const size_t remaining = n - i;
        const size_t count = remaining < lanes ? remaining : lanes;
        auto va = LoadN(d, a + i, count);
        auto vb = LoadN(d, b + i, count);
        sum = MulAdd(va, vb, sum);
        i += count;
    }

    *out = ReduceSum(d, sum);
}


// --- matmul ---

void hwy_matmul(const float* a, const float* b, float* out,
                size_t rows_a, size_t cols_a, size_t cols_b) {
    const ScalableTag<float> d;
    const size_t lanes = Lanes(d);

    for (size_t i = 0; i < rows_a; ++i) {
        size_t j = 0;
        while (j < cols_b) {
            const size_t remaining = cols_b - j;
            const size_t count = remaining < lanes ? remaining : lanes;
            auto acc = Zero(d);

            for (size_t k = 0; k < cols_a; ++k) {
                auto a_broadcast = Set(d, a[i * cols_a + k]);
                auto b_vec = LoadN(d, b + k * cols_b + j, count);
                acc = MulAdd(a_broadcast, b_vec, acc);
            }

            StoreN(acc, d, out + i * cols_b + j, count);
            j += count;
        }
    }
}


// --- matvec ---

void hwy_matvec(const float* a, const float* x, float* out,
                size_t rows, size_t cols) {
    const ScalableTag<float> d;
    const size_t lanes = Lanes(d);

    for (size_t i = 0; i < rows; ++i) {
        auto sum = Zero(d);
        size_t j = 0;

        while (j < cols) {
            const size_t remaining = cols - j;
            const size_t count = remaining < lanes ? remaining : lanes;
            auto a_vec = LoadN(d, a + i * cols + j, count);
            auto x_vec = LoadN(d, x + j, count);
            sum = MulAdd(a_vec, x_vec, sum);
            j += count;
        }

        out[i] = ReduceSum(d, sum);
    }
}

}
}

HWY_AFTER_NAMESPACE();

// --- impl_ wrappers ---

static void impl_add(const float* a, const float* b, float* out, size_t n) {
    hwy::HWY_NAMESPACE::hwy_add(a, b, out, n);
}

static void impl_dot(const float* a, const float* b, float* out, size_t n) {
    hwy::HWY_NAMESPACE::hwy_dot(a, b, out, n);
}

static void impl_matmul(const float* a, const float* b, float* out,
                         size_t rows_a, size_t cols_a, size_t cols_b) {
    hwy::HWY_NAMESPACE::hwy_matmul(a, b, out, rows_a, cols_a, cols_b);
}

static void impl_matvec(const float* a, const float* x, float* out,
                         size_t rows, size_t cols) {
    hwy::HWY_NAMESPACE::hwy_matvec(a, x, out, rows, cols);
}