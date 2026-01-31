#include <cmath>
#include <cstddef>
#include <iostream>

#include "../vector_ops.h"

void test_add_basic() {
    std::cout << "Test 1: Basic add... ";
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[] = {5.0f, 6.0f, 7.0f, 8.0f};
    float out[4] = {};
    vec::add(a, b, out, 4);

    if (fabsf(out[0] - 6.0f) < 1e-4f && fabsf(out[1] - 8.0f) < 1e-4f &&
        fabsf(out[2] - 10.0f) < 1e-4f && fabsf(out[3] - 12.0f) < 1e-4f)
        std::cout << "PASSED\n";
    else
        std::cout << "FAILED\n";
}

void test_add_odd_length() {
    std::cout << "Test 2: Add with odd length (tail handling)... ";
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    float b[] = {7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    float out[7] = {};
    vec::add(a, b, out, 7);

    bool ok = true;
    for (size_t i = 0; i < 7; ++i)
        ok = ok && fabsf(out[i] - 8.0f) < 1e-4f;

    if (ok)
        std::cout << "PASSED\n";
    else
        std::cout << "FAILED\n";
}

void test_add_large() {
    std::cout << "Test 3: Add with 1023 elements... ";
    constexpr size_t N = 1023;
    float a[N], b[N], out[N];
    for (size_t i = 0; i < N; ++i) { a[i] = static_cast<float>(i); b[i] = 1.0f; }
    vec::add(a, b, out, N);

    bool ok = true;
    for (size_t i = 0; i < N; ++i)
        ok = ok && fabsf(out[i] - (static_cast<float>(i) + 1.0f)) < 1e-4f;

    if (ok)
        std::cout << "PASSED\n";
    else
        std::cout << "FAILED\n";
}

void test_dot_basic() {
    std::cout << "Test 4: Basic dot product... ";
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 5.0f, 6.0f};
    float out = 0.0f;
    vec::dot(a, b, &out, 3);

    if (fabsf(out - 32.0f) < 1e-4f)
        std::cout << "PASSED\n";
    else
        std::cout << "FAILED\n";
}

void test_dot_orthogonal() {
    std::cout << "Test 5: Dot product of orthogonal vectors... ";
    float a[] = {1.0f, 0.0f, 0.0f};
    float b[] = {0.0f, 1.0f, 0.0f};
    float out = 99.0f;
    vec::dot(a, b, &out, 3);

    if (fabsf(out) < 1e-4f)
        std::cout << "PASSED\n";
    else
        std::cout << "FAILED\n";
}

void test_dot_large() {
    std::cout << "Test 6: Dot product with 1023 ones... ";
    constexpr size_t N = 1023;
    float a[N], b[N];
    for (size_t i = 0; i < N; ++i) { a[i] = 1.0f; b[i] = 1.0f; }
    float out = 0.0f;
    vec::dot(a, b, &out, N);

    if (fabsf(out - static_cast<float>(N)) < 1e-2f)
        std::cout << "PASSED\n";
    else
        std::cout << "FAILED\n";
}

void test_matmul_identity() {
    std::cout << "Test 7: Matmul with identity matrix... ";
    float eye[] = {1,0, 0,1};
    float a[]   = {5,6, 7,8};
    float out[4] = {};
    vec::matmul(eye, a, out, 2, 2, 2);

    if (fabsf(out[0] - 5) < 1e-4f && fabsf(out[1] - 6) < 1e-4f &&
        fabsf(out[2] - 7) < 1e-4f && fabsf(out[3] - 8) < 1e-4f)
        std::cout << "PASSED\n";
    else
        std::cout << "FAILED\n";
}

void test_matmul_known() {
    std::cout << "Test 8: Matmul 2x2 known result... ";
    // [1 2] * [5 6] = [19 22]
    // [3 4]   [7 8]   [43 50]
    float a[] = {1,2, 3,4};
    float b[] = {5,6, 7,8};
    float out[4] = {};
    vec::matmul(a, b, out, 2, 2, 2);

    if (fabsf(out[0] - 19) < 1e-4f && fabsf(out[1] - 22) < 1e-4f &&
        fabsf(out[2] - 43) < 1e-4f && fabsf(out[3] - 50) < 1e-4f)
        std::cout << "PASSED\n";
    else
        std::cout << "FAILED\n";
}

void test_matmul_nonsquare() {
    std::cout << "Test 9: Matmul non-square (2x3 * 3x2)... ";
    // [1 2 3]   [7  8 ]   [ 58  64]
    // [4 5 6] * [9  10] = [139 154]
    //           [11 12]
    float a[] = {1,2,3, 4,5,6};
    float b[] = {7,8, 9,10, 11,12};
    float out[4] = {};
    vec::matmul(a, b, out, 2, 3, 2);

    if (fabsf(out[0] - 58) < 1e-4f && fabsf(out[1] - 64) < 1e-4f &&
        fabsf(out[2] - 139) < 1e-4f && fabsf(out[3] - 154) < 1e-4f)
        std::cout << "PASSED\n";
    else
        std::cout << "FAILED\n";
}

void test_matmul_row_col() {
    std::cout << "Test 10: Matmul row * column (1x4 * 4x1)... ";
    float a[] = {1, 2, 3, 4};
    float b[] = {5, 6, 7, 8};
    float out[1] = {};
    vec::matmul(a, b, out, 1, 4, 1);

    if (fabsf(out[0] - 70) < 1e-4f)
        std::cout << "PASSED\n";
    else
        std::cout << "FAILED\n";
}

void test_matvec_identity() {
    std::cout << "Test 11: Matvec with identity matrix... ";
    float eye[] = {1,0,0, 0,1,0, 0,0,1};
    float x[] = {3, 5, 7};
    float out[3] = {};
    vec::matvec(eye, x, out, 3, 3);

    if (fabsf(out[0] - 3) < 1e-4f && fabsf(out[1] - 5) < 1e-4f &&
        fabsf(out[2] - 7) < 1e-4f)
        std::cout << "PASSED\n";
    else
        std::cout << "FAILED\n";
}

void test_matvec_known() {
    std::cout << "Test 12: Matvec 2x2 known result... ";
    // [1 2] * [5] = [17]
    // [3 4]   [6]   [39]
    float a[] = {1,2, 3,4};
    float x[] = {5, 6};
    float out[2] = {};
    vec::matvec(a, x, out, 2, 2);

    if (fabsf(out[0] - 17) < 1e-4f && fabsf(out[1] - 39) < 1e-4f)
        std::cout << "PASSED\n";
    else
        std::cout << "FAILED\n";
}

void test_matvec_nonsquare() {
    std::cout << "Test 13: Matvec non-square (2x3)... ";
    // [1 2 3] * [1] = [14]
    // [4 5 6]   [2]   [32]
    //           [3]
    float a[] = {1,2,3, 4,5,6};
    float x[] = {1, 2, 3};
    float out[2] = {};
    vec::matvec(a, x, out, 2, 3);

    if (fabsf(out[0] - 14) < 1e-4f && fabsf(out[1] - 32) < 1e-4f)
        std::cout << "PASSED\n";
    else
        std::cout << "FAILED\n";
}

void test_matvec_wide() {
    std::cout << "Test 14: Matvec wide row (1x7, tail handling)... ";
    float a[] = {1,1,1,1,1,1,1};
    float x[] = {1,2,3,4,5,6,7};
    float out[1] = {};
    vec::matvec(a, x, out, 1, 7);

    if (fabsf(out[0] - 28) < 1e-4f)
        std::cout << "PASSED\n";
    else
        std::cout << "FAILED\n";
}

int main() {
    std::cout << "=== Vector Ops Test Suite ===\n\n";

    test_add_basic();
    test_add_odd_length();
    test_add_large();

    test_dot_basic();
    test_dot_orthogonal();
    test_dot_large();

    test_matmul_identity();
    test_matmul_known();
    test_matmul_nonsquare();
    test_matmul_row_col();

    test_matvec_identity();
    test_matvec_known();
    test_matvec_nonsquare();
    test_matvec_wide();

    std::cout << "\n=== All tests complete ===\n";
    return 0;
}
