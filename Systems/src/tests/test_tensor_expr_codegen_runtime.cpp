#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <cstdio>
#include <utility>

#include "Systems/src/ExprSystem/ExprFunctions.h"
#include "Systems/src/tensor.h"

using Vec64 = TensorBase<float, std::extents<size_t, 64>>;

float eval_kernel(const Vec64& a, const Vec64& b, const Vec64& c, Vec64& out) {
    in_place_eval(a + b * c, out);
    float sum = 0.0f;
    for (size_t i = 0; i < Vec64::iter_size(); ++i) sum += out.access(i);
    return sum;
}

int main() {
    std::array<float, Vec64::iter_size()> a_data{};
    std::array<float, Vec64::iter_size()> b_data{};
    std::array<float, Vec64::iter_size()> c_data{};

    unsigned seed = 12345u;
    for (size_t i = 0; i < Vec64::iter_size(); ++i) {
        seed = seed * 1664525u + 1013904223u;
        a_data[i] = static_cast<float>(seed & 0x3ffu) - 500.0f;
        seed = seed * 1664525u + 1013904223u;
        b_data[i] = static_cast<float>(seed & 0x3ffu) * 0.125f;
        seed = seed * 1664525u + 1013904223u;
        c_data[i] = static_cast<float>(seed & 0x1ffu) * 0.25f;
    }

    const Vec64 a(a_data.data());
    const Vec64 b(b_data.data());
    const Vec64 c(c_data.data());
    Vec64 out;

    float checksum = 0.0f;
    for (int rep = 0; rep < 1000; ++rep) checksum += eval_kernel(a, b, c, out);
    std::printf("%f\n", checksum);
    return 0;
}
