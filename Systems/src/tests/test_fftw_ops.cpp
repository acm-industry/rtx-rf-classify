#include <iostream>
#include <complex>
#include <vector>
#include <cmath>

#include "../fft/fftw_ops.h"
#include "../memorybuffer.h"
#include "../tensor.h"
#include "../ExprSystem/Expression.h"

using namespace fft;

int passed = 0, total = 0;

#define TEST(name, condition) \
    total++; \
    if (condition) { \
        passed++; \
        std::cout << "✓ " << name << "\n"; \
    } else { \
        std::cout << "✗ " << name << "\n"; \
    }

bool approx_equal(double a, double b, double epsilon = 1e-5) {
    return std::abs(a - b) < epsilon;
}

void test_basic_plan() {
    using Extent = std::extents<size_t, 8>;
    using T = std::complex<float>;

    DynTensor<T, Extent> in;
    DynTensor<T, Extent> out;

    for (size_t i = 0; i < 8; ++i) {
        in.flat(i) = T(static_cast<float>(i), 0.0f);
    }

    FFTW<float, Extent, FFTW_FORWARD, FFTW_ESTIMATE> fft;
    fft(in, out);

    // DC component (sum of inputs) should be 28
    TEST("Basic Plan DC Real", approx_equal(out.flat(0).real(), 28.0f));
    TEST("Basic Plan DC Imag", approx_equal(out.flat(0).imag(), 0.0f));
}

template<class T, class Extent>
struct DummyExpr {
    static constexpr auto extents = Extent{};
    using value_type = T;
    static constexpr size_t iter_size() noexcept { return 4; }
    
    constexpr T access(size_t i) const noexcept {
        return T(static_cast<double>(i * 2), 0.0);
    }
};

void test_eval_integration() {
    using Extent = std::extents<size_t, 4>;
    using T = std::complex<double>;

    DynTensor<T, Extent> out;
    DynTensor<T, Extent> temp_in;

    DummyExpr<T, Extent> expr;
    FFTW<double, Extent, FFTW_FORWARD, FFTW_ESTIMATE> fft;
    
    fft.eval(expr, temp_in, out);

    // Sum is 0 + 2 + 4 + 6 = 12
    TEST("Eval Integration DC Real", approx_equal(out.flat(0).real(), 12.0));
    TEST("Eval Integration DC Imag", approx_equal(out.flat(0).imag(), 0.0));
}

void test_custom_allocator() {
    using Extent = std::extents<size_t, 16>;
    using T = std::complex<float>;

    MemoryBuffer buf(1024);
    auto alloc = buf.get_allocator<T, 32>();

    DynTensor<T, Extent, decltype(alloc)> in(alloc);
    DynTensor<T, Extent, decltype(alloc)> out(alloc);

    for (size_t i = 0; i < 16; ++i) {
        in.flat(i) = T(1.0f, 0.0f);
    }

    FFTW<float, Extent> fft;
    fft(in, out);

    // DC is 16, others are 0
    TEST("Custom Allocator DC Real", approx_equal(out.flat(0).real(), 16.0f));
    TEST("Custom Allocator DC Imag", approx_equal(out.flat(0).imag(), 0.0f));
    
    bool others_zero = true;
    for (size_t i = 1; i < 16; ++i) {
        if (!approx_equal(out.flat(i).real(), 0.0f) || !approx_equal(out.flat(i).imag(), 0.0f)) {
            others_zero = false;
        }
    }
    TEST("Custom Allocator Others Zero", others_zero);
}

int main() {
    std::cout << "\n=== FFTW Wrapper Tests ===\n\n";
    
    test_basic_plan();
    test_eval_integration();
    test_custom_allocator();
    
    std::cout << "\n" << passed << "/" << total << " tests passed\n\n";
    return (passed == total) ? 0 : 1;
}
