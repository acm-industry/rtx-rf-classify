#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>

#include "../ExprSystem/ExprFunctions.h"
#include "../ExprSystem/Expression.h"
#include "../tensor.h"

using Vec16 = TensorBase<float, std::extents<size_t, 16>>;

static_assert(Expression<Vec16>, "TensorBase must satisfy Expression");

constexpr bool nearly_equal(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) < eps;
}

void test_linear_access() {
    std::array<float, Vec16::iter_size()> seed{};
    for (size_t i = 0; i < seed.size(); ++i)
        seed[i] = static_cast<float>(i);

    Vec16 tensor(seed.data());
    for (size_t i = 0; i < Vec16::iter_size(); ++i) {
        assert(nearly_equal(tensor.access(i), seed[i]));
        assert(nearly_equal(tensor.flat(i), seed[i]));
    }

    tensor.access(3) = 99.25f;
    assert(nearly_equal(tensor.flat(3), 99.25f));
}

void test_expr_eval() {
    std::array<float, Vec16::iter_size()> a_data{};
    std::array<float, Vec16::iter_size()> b_data{};
    std::array<float, Vec16::iter_size()> c_data{};

    for (size_t i = 0; i < Vec16::iter_size(); ++i) {
        a_data[i] = static_cast<float>(i) - 3.0f;
        b_data[i] = static_cast<float>(i) * 0.5f;
        c_data[i] = 1.0f + static_cast<float>(i % 3);
    }

    Vec16 a(a_data.data());
    Vec16 b(b_data.data());
    Vec16 c(c_data.data());
    Vec16 out;

    in_place_eval(a + b * c, out);
    for (size_t i = 0; i < Vec16::iter_size(); ++i) {
        const float expected = a_data[i] + (b_data[i] * c_data[i]);
        assert(nearly_equal(out.access(i), expected));
    }

    in_place_eval(relu(a - b), out);
    for (size_t i = 0; i < Vec16::iter_size(); ++i) {
        const float expected = std::max(0.0f, a_data[i] - b_data[i]);
        assert(nearly_equal(out.access(i), expected));
    }
}

int main() {
    test_linear_access();
    test_expr_eval();
    std::cout << "test_tensor_expr: PASS\n";
    return 0;
}
