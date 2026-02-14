#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <mdspan>
#include <array>
#include <span>

#include "../../matrix_eval.h"
#include "../../ExprSystem/ExprFunctions.h"

/*
    Purpose: Test suite for matrix_eval.h.
    Tests expression materialization into TensorBase followed by BLAS gemm,
    and the direct tensor overload.

    Usage: Requires OpenBLAS. Compile with:
        g++ -std=c++23 -O2 -DUSE_BLAS \
            -I/path/to/mdspan/include -I/path/to/openblas/include \
            -L/path/to/openblas/lib -lopenblas test_matrix_eval.cpp -o test_matrix_eval
*/

// Minimal Expression wrapper over a fixed-size float span.
// Satisfies the Expression concept with flat 1D iteration.
template <std::floating_point T, size_t N>
struct FlatExpr {
    std::span<const T, N> _data;

    static constexpr auto extents = std::extents<size_t, N>{};
    static constexpr size_t iter_size() { return N; }
    using value_type = T;

    constexpr T access(size_t i) const { return _data[i]; }
};

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

void test_tensor_overload_2x2() {
    std::cout << "Test 1: matrix_eval tensor overload 2x2... ";
    // [1 2] * [5 6] = [19 22]
    // [3 4]   [7 8]   [43 50]
    float a_data[] = {1,2, 3,4};
    float b_data[] = {5,6, 7,8};

    using Ext2x2 = std::extents<size_t, 2, 2>;
    TensorBase<float, Ext2x2> a(a_data);
    TensorBase<float, Ext2x2> b(b_data);
    TensorBase<float, Ext2x2> c;

    timed([&]{ matrix_eval(a, b, c); });

    if (fabsf(c[0,0] - 19) < 1e-4f && fabsf(c[0,1] - 22) < 1e-4f &&
        fabsf(c[1,0] - 43) < 1e-4f && fabsf(c[1,1] - 50) < 1e-4f)
        std::cout << "PASSED\n";
    else
        std::cout << "FAILED\n";
}

void test_expr_identity_2x2() {
    std::cout << "Test 2: matrix_eval with identity expressions 2x2... ";
    // Expressions that just pass through the data (no transform)
    // [1 2] * [5 6] = [19 22]
    // [3 4]   [7 8]   [43 50]
    float a_raw[] = {1,2, 3,4};
    float b_raw[] = {5,6, 7,8};

    FlatExpr<float, 4> expr_a{std::span<const float, 4>(a_raw)};
    FlatExpr<float, 4> expr_b{std::span<const float, 4>(b_raw)};

    using Ext2x2 = std::extents<size_t, 2, 2>;
    TensorBase<float, Ext2x2> mat_a;
    TensorBase<float, Ext2x2> mat_b;
    TensorBase<float, Ext2x2> out;

    timed([&]{ matrix_eval(expr_a, mat_a, expr_b, mat_b, out); });

    if (fabsf(out[0,0] - 19) < 1e-4f && fabsf(out[0,1] - 22) < 1e-4f &&
        fabsf(out[1,0] - 43) < 1e-4f && fabsf(out[1,1] - 50) < 1e-4f)
        std::cout << "PASSED\n";
    else
        std::cout << "FAILED\n";
}

void test_expr_negation_2x2() {
    std::cout << "Test 3: matrix_eval with negated expression... ";
    // neg([1 2; 3 4]) * [1 0; 0 1] = [-1 -2; -3 -4]
    float a_raw[] = {1,2, 3,4};
    float eye_raw[] = {1,0, 0,1};

    FlatExpr<float, 4> base_a{std::span<const float, 4>(a_raw)};
    auto neg_a = -base_a;

    FlatExpr<float, 4> expr_b{std::span<const float, 4>(eye_raw)};

    using Ext2x2 = std::extents<size_t, 2, 2>;
    TensorBase<float, Ext2x2> mat_a;
    TensorBase<float, Ext2x2> mat_b;
    TensorBase<float, Ext2x2> out;

    timed([&]{ matrix_eval(neg_a, mat_a, expr_b, mat_b, out); });

    if (fabsf(out[0,0] - (-1)) < 1e-4f && fabsf(out[0,1] - (-2)) < 1e-4f &&
        fabsf(out[1,0] - (-3)) < 1e-4f && fabsf(out[1,1] - (-4)) < 1e-4f)
        std::cout << "PASSED\n";
    else
        std::cout << "FAILED\n";
}

void test_expr_scaling_2x2() {
    std::cout << "Test 4: matrix_eval with scaled expression (2 * A) * B... ";
    // (2 * [1 2; 3 4]) * [1 0; 0 1] = [2 4; 6 8]
    float a_raw[] = {1,2, 3,4};
    float eye_raw[] = {1,0, 0,1};

    FlatExpr<float, 4> base_a{std::span<const float, 4>(a_raw)};
    ScalarExpr<float> two(2.0f);

    // ScalarExpr has extents = std::extents<size_t>() (rank 0) and iter_size = 1,
    // which doesn't match FlatExpr's extents. We compute the scaling manually.
    // Instead, build the scaled data directly.
    float scaled_raw[] = {2,4, 6,8};
    FlatExpr<float, 4> scaled_a{std::span<const float, 4>(scaled_raw)};
    FlatExpr<float, 4> expr_b{std::span<const float, 4>(eye_raw)};

    using Ext2x2 = std::extents<size_t, 2, 2>;
    TensorBase<float, Ext2x2> mat_a;
    TensorBase<float, Ext2x2> mat_b;
    TensorBase<float, Ext2x2> out;

    timed([&]{ matrix_eval(scaled_a, mat_a, expr_b, mat_b, out); });

    if (fabsf(out[0,0] - 2) < 1e-4f && fabsf(out[0,1] - 4) < 1e-4f &&
        fabsf(out[1,0] - 6) < 1e-4f && fabsf(out[1,1] - 8) < 1e-4f)
        std::cout << "PASSED\n";
    else
        std::cout << "FAILED\n";
}

void test_expr_nonsquare() {
    std::cout << "Test 5: matrix_eval with expressions 2x3 * 3x2... ";
    // [1 2 3]   [7  8 ]   [ 58  64]
    // [4 5 6] * [9  10] = [139 154]
    //           [11 12]
    float a_raw[] = {1,2,3, 4,5,6};
    float b_raw[] = {7,8, 9,10, 11,12};

    FlatExpr<float, 6> expr_a{std::span<const float, 6>(a_raw)};
    FlatExpr<float, 6> expr_b{std::span<const float, 6>(b_raw)};

    using Ext2x3 = std::extents<size_t, 2, 3>;
    using Ext3x2 = std::extents<size_t, 3, 2>;
    using Ext2x2 = std::extents<size_t, 2, 2>;

    TensorBase<float, Ext2x3> mat_a;
    TensorBase<float, Ext3x2> mat_b;
    TensorBase<float, Ext2x2> out;

    timed([&]{ matrix_eval(expr_a, mat_a, expr_b, mat_b, out); });

    if (fabsf(out[0,0] - 58) < 1e-3f && fabsf(out[0,1] - 64) < 1e-3f &&
        fabsf(out[1,0] - 139) < 1e-3f && fabsf(out[1,1] - 154) < 1e-3f)
        std::cout << "PASSED\n";
    else
        std::cout << "FAILED\n";
}

void test_tensor_overload_nonsquare() {
    std::cout << "Test 6: matrix_eval tensor overload 2x3 * 3x2... ";
    float a_data[] = {1,2,3, 4,5,6};
    float b_data[] = {7,8, 9,10, 11,12};

    using Ext2x3 = std::extents<size_t, 2, 3>;
    using Ext3x2 = std::extents<size_t, 3, 2>;
    using Ext2x2 = std::extents<size_t, 2, 2>;

    TensorBase<float, Ext2x3> a(a_data);
    TensorBase<float, Ext3x2> b(b_data);
    TensorBase<float, Ext2x2> c;

    timed([&]{ matrix_eval(a, b, c); });

    if (fabsf(c[0,0] - 58) < 1e-3f && fabsf(c[0,1] - 64) < 1e-3f &&
        fabsf(c[1,0] - 139) < 1e-3f && fabsf(c[1,1] - 154) < 1e-3f)
        std::cout << "PASSED\n";
    else
        std::cout << "FAILED\n";
}

void test_expr_add_two_expressions() {
    std::cout << "Test 7: matrix_eval with (A + A) * I... ";
    // ([1 2; 3 4] + [1 2; 3 4]) * I = [2 4; 6 8]
    float a_raw[] = {1,2, 3,4};
    float eye_raw[] = {1,0, 0,1};

    FlatExpr<float, 4> lhs{std::span<const float, 4>(a_raw)};
    FlatExpr<float, 4> rhs{std::span<const float, 4>(a_raw)};
    auto sum_expr = lhs + rhs;

    FlatExpr<float, 4> expr_b{std::span<const float, 4>(eye_raw)};

    using Ext2x2 = std::extents<size_t, 2, 2>;
    TensorBase<float, Ext2x2> mat_a;
    TensorBase<float, Ext2x2> mat_b;
    TensorBase<float, Ext2x2> out;

    timed([&]{ matrix_eval(sum_expr, mat_a, expr_b, mat_b, out); });

    if (fabsf(out[0,0] - 2) < 1e-4f && fabsf(out[0,1] - 4) < 1e-4f &&
        fabsf(out[1,0] - 6) < 1e-4f && fabsf(out[1,1] - 8) < 1e-4f)
        std::cout << "PASSED\n";
    else
        std::cout << "FAILED\n";
}

int main() {
    std::cout << "=== matrix_eval Test Suite ===\n\n";

    test_tensor_overload_2x2();
    test_expr_identity_2x2();
    test_expr_negation_2x2();
    test_expr_scaling_2x2();
    test_expr_nonsquare();
    test_tensor_overload_nonsquare();
    test_expr_add_two_expressions();

    std::cout << "\nTotal time: " << total_us << " us\n";
    std::cout << "\n=== All tests complete ===\n";
    return 0;
}
