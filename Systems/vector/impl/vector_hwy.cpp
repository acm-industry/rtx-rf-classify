#include "../vector_ops.h"
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();

namespace hwy {
namespace HWY_NAMESPACE {

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

}
}

HWY_AFTER_NAMESPACE();

static void impl_add(const float* a, const float* b, float* out, size_t n) {
    hwy::HWY_NAMESPACE::hwy_add(a, b, out, n);
}

static void impl_dot(const float* a, const float* b, float* out, size_t n) {
    hwy::HWY_NAMESPACE::hwy_dot(a, b, out, n);
}
