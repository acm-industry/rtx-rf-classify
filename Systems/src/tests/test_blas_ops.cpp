#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <mdspan>

#include "../blas_ops.h"

/*
    Purpose: Test suite for blas_ops.h mdspan BLAS wrappers.
    Tests gemm, gemv, and dot for both float and double types.

    Usage: Requires OpenBLAS. Compile with:
        g++ -std=c++23 -O2 -I/path/to/mdspan/include -I/path/to/openblas/include \
            -L/path/to/openblas/lib -lopenblas test_blas_ops.cpp -o test_blas_ops
*/

static double total_us = 0.0;

template<typename F>
void timed(F&& fn) {
    auto start = std::chrono::high_resolution_clock::now();
    fn();
    auto end = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(end - start).count();
    total_us += us;
    std::cout << " (" << us << " us)  ";
}

void test_gemm_float_2x2() {
    std::cout << "Test 1: gemm float 2x2 known result... ";
    // [1 2] * [5 6] = [19 22]
    // [3 4]   [7 8]   [43 50]
    float a_data[] = {1,2, 3,4};
    float b_data[] = {5,6, 7,8};
    float c_data[4] = {};

    using ext_2x2 = std::extents<size_t, 2, 2>;
    std::mdspan<float, ext_2x2> a(a_data);
    std::mdspan<float, ext_2x2> b(b_data);
    std::mdspan<float, ext_2x2> c(c_data);

    timed([&]{ blas::gemm(a, b, c); });

    if (fabsf(c_data[0] - 19) < 1e-4f && fabsf(c_data[1] - 22) < 1e-4f &&
        fabsf(c_data[2] - 43) < 1e-4f && fabsf(c_data[3] - 50) < 1e-4f)
        std::cout << "PASSED\n";
    else
        std::cout << "FAILED\n";
}

void test_gemm_float_identity() {
    std::cout << "Test 2: gemm float with identity matrix... ";
    float eye[] = {1,0, 0,1};
    float a_data[] = {5,6, 7,8};
    float c_data[4] = {};

    using ext_2x2 = std::extents<size_t, 2, 2>;
    std::mdspan<float, ext_2x2> eye_m(eye);
    std::mdspan<float, ext_2x2> a(a_data);
    std::mdspan<float, ext_2x2> c(c_data);

    timed([&]{ blas::gemm(eye_m, a, c); });

    if (fabsf(c_data[0] - 5) < 1e-4f && fabsf(c_data[1] - 6) < 1e-4f &&
        fabsf(c_data[2] - 7) < 1e-4f && fabsf(c_data[3] - 8) < 1e-4f)
        std::cout << "PASSED\n";
    else
        std::cout << "FAILED\n";
}

void test_gemm_double_nonsquare() {
    std::cout << "Test 3: gemm double 2x3 * 3x2... ";
    // [1 2 3]   [7  8 ]   [ 58  64]
    // [4 5 6] * [9  10] = [139 154]
    //           [11 12]
    double a_data[] = {1,2,3, 4,5,6};
    double b_data[] = {7,8, 9,10, 11,12};
    double c_data[4] = {};

    using ext_2x3 = std::extents<size_t, 2, 3>;
    using ext_3x2 = std::extents<size_t, 3, 2>;
    using ext_2x2 = std::extents<size_t, 2, 2>;

    std::mdspan<double, ext_2x3> a(a_data);
    std::mdspan<double, ext_3x2> b(b_data);
    std::mdspan<double, ext_2x2> c(c_data);

    timed([&]{ blas::gemm(a, b, c); });

    if (fabs(c_data[0] - 58) < 1e-10 && fabs(c_data[1] - 64) < 1e-10 &&
        fabs(c_data[2] - 139) < 1e-10 && fabs(c_data[3] - 154) < 1e-10)
        std::cout << "PASSED\n";
    else
        std::cout << "FAILED\n";
}

void test_gemv_float() {
    std::cout << "Test 4: gemv float 2x2... ";
    // [1 2] * [5] = [17]
    // [3 4]   [6]   [39]
    float a_data[] = {1,2, 3,4};
    float x_data[] = {5, 6};
    float y_data[2] = {};

    using ext_2x2 = std::extents<size_t, 2, 2>;
    using ext_2 = std::extents<size_t, 2>;

    std::mdspan<float, ext_2x2> a(a_data);
    std::mdspan<float, ext_2> x(x_data);
    std::mdspan<float, ext_2> y(y_data);

    timed([&]{ blas::gemv(a, x, y); });

    if (fabsf(y_data[0] - 17) < 1e-4f && fabsf(y_data[1] - 39) < 1e-4f)
        std::cout << "PASSED\n";
    else
        std::cout << "FAILED\n";
}

void test_gemv_double_nonsquare() {
    std::cout << "Test 5: gemv double 2x3... ";
    // [1 2 3] * [1] = [14]
    // [4 5 6]   [2]   [32]
    //           [3]
    double a_data[] = {1,2,3, 4,5,6};
    double x_data[] = {1, 2, 3};
    double y_data[2] = {};

    using ext_2x3 = std::extents<size_t, 2, 3>;
    using ext_3 = std::extents<size_t, 3>;
    using ext_2 = std::extents<size_t, 2>;

    std::mdspan<double, ext_2x3> a(a_data);
    std::mdspan<double, ext_3> x(x_data);
    std::mdspan<double, ext_2> y(y_data);

    timed([&]{ blas::gemv(a, x, y); });

    if (fabs(y_data[0] - 14) < 1e-10 && fabs(y_data[1] - 32) < 1e-10)
        std::cout << "PASSED\n";
    else
        std::cout << "FAILED\n";
}

void test_dot_float() {
    std::cout << "Test 6: dot float... ";
    float a_data[] = {1, 2, 3};
    float b_data[] = {4, 5, 6};

    using ext_3 = std::extents<size_t, 3>;
    std::mdspan<float, ext_3> a(a_data);
    std::mdspan<float, ext_3> b(b_data);

    float result;
    timed([&]{ result = blas::dot(a, b); });

    if (fabsf(result - 32.0f) < 1e-4f)
        std::cout << "PASSED\n";
    else
        std::cout << "FAILED\n";
}

void test_dot_double() {
    std::cout << "Test 7: dot double... ";
    double a_data[] = {1, 2, 3, 4};
    double b_data[] = {5, 6, 7, 8};

    using ext_4 = std::extents<size_t, 4>;
    std::mdspan<double, ext_4> a(a_data);
    std::mdspan<double, ext_4> b(b_data);

    double result;
    timed([&]{ result = blas::dot(a, b); });

    // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    if (fabs(result - 70.0) < 1e-10)
        std::cout << "PASSED\n";
    else
        std::cout << "FAILED\n";
}

void test_dot_orthogonal() {
    std::cout << "Test 8: dot float orthogonal vectors... ";
    float a_data[] = {1, 0, 0};
    float b_data[] = {0, 1, 0};

    using ext_3 = std::extents<size_t, 3>;
    std::mdspan<float, ext_3> a(a_data);
    std::mdspan<float, ext_3> b(b_data);

    float result;
    timed([&]{ result = blas::dot(a, b); });

    if (fabsf(result) < 1e-4f)
        std::cout << "PASSED\n";
    else
        std::cout << "FAILED\n";
}

int main() {
    std::cout << "=== BLAS Ops (mdspan) Test Suite ===\n\n";

    test_gemm_float_2x2();
    test_gemm_float_identity();
    test_gemm_double_nonsquare();

    test_gemv_float();
    test_gemv_double_nonsquare();

    test_dot_float();
    test_dot_double();
    test_dot_orthogonal();

    std::cout << "\nTotal time: " << total_us << " us\n";
    std::cout << "\n=== All tests complete ===\n";
    return 0;
}
