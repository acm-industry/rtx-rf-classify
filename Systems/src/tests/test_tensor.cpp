#include <iostream>
#include "tensor.h"

using Matrix3x3 = TensorBase<double, std::extents<size_t, 3, 3>>;
using Vector5 = TensorBase<double, std::extents<size_t, 5>>;

int passed = 0, total = 0;

#define TEST(name, condition) \
    total++; \
    if (condition) { \
        passed++; \
        std::cout << "✓ " << name << "\n"; \
    } else { \
        std::cout << "✗ " << name << "\n"; \
    }

int main() {
    std::cout << "\n=== TensorBase Tests ===\n\n";
    
    // Test 1: Matrix multiplication
    double A_w[9] = {1,2,3,4,5,6,7,8,9};
    double B_w[9] = {9,8,7,6,5,4,3,2,1};
    Matrix3x3 A(A_w), B(B_w);
    double C_w[9] = {0};
    Matrix3x3 C(C_w);
    
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 3; k++) {
                C(i,j) += A(i,k) * B(k,j);
            }
        }
    }
    TEST("Matrix multiplication", C(0,0) == 30 && C(1,1) == 69);
    
    // Test 2: Copy
    Matrix3x3 A2 = A;
    A(0,0) = 999;
    TEST("Deep copy", A2(0,0) == 1.0);
    
    // Test 3: Move
    double* ptr = A2.data();
    Matrix3x3 A3 = std::move(A2);
    TEST("Move", A3.data() == ptr && A2.data() == nullptr);
    
    // Test 4: Addition
    double M1_w[9] = {1,2,3,4,5,6,7,8,9};
    double M2_w[9] = {9,8,7,6,5,4,3,2,1};
    Matrix3x3 M1(M1_w), M2(M2_w), M3;
    for (size_t i = 0; i < 3; i++)
        for (size_t j = 0; j < 3; j++)
            M3(i,j) = M1(i,j) + M2(i,j);
    TEST("Matrix addition", M3(0,0) == 10 && M3(2,2) == 10);
    
    // Test 5: Transpose
    Matrix3x3 T;
    for (size_t i = 0; i < 3; i++)
        for (size_t j = 0; j < 3; j++)
            T(i,j) = M1(j,i);
    TEST("Transpose", T(0,1) == 4 && T(1,0) == 2);
    
    // Test 6: Scalar multiply
    for (size_t i = 0; i < 3; i++)
        for (size_t j = 0; j < 3; j++)
            M1(i,j) *= 2;
    TEST("Scalar multiply", M1(0,0) == 2 && M1(2,2) == 18);
    
    // Test 7: Vector dot product
    double v1_w[5] = {1,2,3,4,5};
    double v2_w[5] = {5,4,3,2,1};
    Vector5 v1(v1_w), v2(v2_w);
    double dot = 0;
    for (size_t i = 0; i < 5; i++)
        dot += v1(i) * v2(i);
    TEST("Dot product", dot == 35);
    
    // Test 8: Flat access
    TEST("Flat access", A3.flat(4) == 5 && A3.flat(8) == 9);
    
    // Test 9: Zero init
    Matrix3x3 Z;
    bool zero = true;
    for (size_t i = 0; i < 9; i++)
        if (Z.flat(i) != 0) zero = false;
    TEST("Zero init", zero);
    
    std::cout << "\n" << passed << "/" << total << " tests passed\n\n";
    return (passed == total) ? 0 : 1;
}