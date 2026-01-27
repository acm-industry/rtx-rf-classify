#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();

namespace hwy {
namespace HWY_NAMESPACE {


// element-wise addition of two float arrays
void add(const float* a, const float* b, float* out, size_t n) {
    const ScalableTag<float> d;

    size_t i = 0;
    while (i < n) {
        auto va = Load(d, a + i);
        auto vb = Load(d, b + i);
        Store(Add(va, vb), d, out + i);
        i += Lanes(d);
    }
}


// dot product of two float arrays
void dot(const float* a, const float* b, float* out, size_t n) {
    const ScalableTag<float> d;
    auto sum = Zero(d);

    size_t i = 0;
    while (i < n) { // reason no for for loop is to handle tail 
        auto va = Load(d, a + i);
        auto vb = Load(d, b + i);
        sum = Add(sum, Mul(va, vb));
        i += Lanes(d);
    }

    *out = ReduceSum(d, sum);
}


}
}

HWY_AFTER_NAMESPACE();