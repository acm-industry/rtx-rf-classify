#include "fp16_decode.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

static void expect_near(float a, float b, const char* ctx) {
    if (std::isnan(a) && std::isnan(b)) {
        return;
    }
    const float tol = 1e-5f;
    if (std::abs(a - b) > tol * (1.f + std::abs(b))) {
        std::cerr << "FAIL " << ctx << " got " << a << " expected " << b << '\n';
        std::exit(1);
    }
}

int main() {
    using rtx::fp16::half_to_float;

    expect_near(half_to_float(0x0000), 0.f, "zero");
    expect_near(half_to_float(0x0001), 5.96046448e-08f, "smallest subnormal");
    expect_near(half_to_float(0x0400), 6.10351562e-05f, "normal small");
    expect_near(half_to_float(0x3c00), 1.f, "one");
    expect_near(half_to_float(0xbc00), -1.f, "minus one");
    expect_near(half_to_float(0x3555), 0.333251953125f, "fraction");

    const float inf = half_to_float(0x7c00);
    if (!std::isinf(inf) || inf < 0.f) {
        std::cerr << "FAIL +inf\n";
        return 1;
    }
    const float ninf = half_to_float(0xfc00);
    if (!std::isinf(ninf) || ninf > 0.f) {
        std::cerr << "FAIL -inf\n";
        return 1;
    }
    const float nanv = half_to_float(0x7e00);
    if (!std::isnan(nanv)) {
        std::cerr << "FAIL nan\n";
        return 1;
    }

    std::cout << "test_fp16_decode ok\n";
    return 0;
}
