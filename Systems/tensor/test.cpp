#include <iostream>
#include "tensor.cpp"

using Matrix3x3 = TensorBase<double, std::extents<size_t, 3, 3>>;

int main() {
    // Example weights for Matrix A
    double A_weights[9] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0
    };

    // Example weights for Matrix B
    double B_weights[9] = {
        9.0, 8.0, 7.0,
        6.0, 5.0, 4.0,
        3.0, 2.0, 1.0
    };

    // Create TensorBase matrices
    Matrix3x3 A(A_weights);
    Matrix3x3 B(B_weights);

    // Prepare C to hold the result
    double C_weights[9] = {0.0}; // initialized to 0
    Matrix3x3 C(C_weights);      // allocate and copy initial zeros

    // Matrix multiplication: C = A * B
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < 3; ++k) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }

    // Print result
    std::cout << "C = A * B:\n";
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            std::cout << C(i, j) << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
