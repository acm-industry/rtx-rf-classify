#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>

#include "../ExprSystem/ExprFunctions.h"
#include "../tensor.h"

using Vec16 = TensorBase<float, std::extents<size_t, 16>>;

int main() {
    constexpr std::array<float, 16> left = {0.0f,  1.0f,  2.0f,  3.0f,
                                            4.0f,  5.0f,  6.0f,  7.0f,
                                            8.0f,  9.0f,  10.0f, 11.0f,
                                            12.0f, 13.0f, 14.0f, 15.0f};
    constexpr std::array<float, 16> right = {10.0f, 10.0f, 10.0f, 10.0f,
                                             10.0f, 10.0f, 10.0f, 10.0f,
                                             10.0f, 10.0f, 10.0f, 10.0f,
                                             10.0f, 10.0f, 10.0f, 10.0f};

    const Vec16 a(left.data());
    const Vec16 b(right.data());
    Vec16 out;

    in_place_eval(a + b, out);

    float checksum = 0.0f;
    for (size_t i = 0; i < Vec16::iter_size(); ++i) checksum += out.access(i);

    assert(std::fabs(checksum - 280.0f) < 1e-4f);
    std::cout << "test_tensor_expr_constdata: PASS\n";
    return 0;
}
