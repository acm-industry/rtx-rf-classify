#include "../src/memorybuffer.h"
#include "tensor.h"
#include <iostream>

// Type aliases for convenience with MemoryBuffer allocator
template<typename T, FixedExtent E>
using PoolTensor = TensorBase<T, E, MemoryBuffer::Allocator<T>>;

using PoolMatrix3x3 = PoolTensor<float, std::extents<std::size_t, 3, 3>>;
using PoolVector5 = PoolTensor<float, std::extents<std::size_t, 5>>;

void print_matrix(const auto& mat, const char* name) {
    std::cout << name << ":\n";
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            std::cout << mat(i, j) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    std::cout << "========================================\n";
    std::cout << "  TensorBase with MemoryBuffer Demo\n";
    std::cout << "========================================\n\n";

    // ===== EXAMPLE 1: Basic Memory Pool Usage =====
    std::cout << "EXAMPLE 1: Creating tensors from a memory pool\n";
    std::cout << "-----------------------------------------------\n";
    
    // Calculate memory needed
    std::size_t matrix_size = 3 * 3 * sizeof(float);
    std::size_t vector_size = 5 * sizeof(float);
    std::size_t total_needed = 2 * matrix_size + vector_size + 256; // Extra padding
    
    std::cout << "Memory pool size: " << total_needed << " bytes\n\n";
    
    // Create memory pool
    MemoryBuffer pool(total_needed);
    
    // Get allocator from pool
    auto alloc = pool.get_allocator<float>();
    
    // Create tensors using the pool
    PoolMatrix3x3 A(alloc);
    PoolMatrix3x3 B(alloc);
    PoolVector5 v(alloc);
    
    // Fill with data
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            A(i, j) = static_cast<float>(i * 3 + j + 1);
            B(i, j) = static_cast<float>(9 - (i * 3 + j));
        }
    }
    
    for (std::size_t i = 0; i < 5; ++i) {
        v(i) = static_cast<float>(i + 1);
    }
    
    print_matrix(A, "Matrix A");
    print_matrix(B, "Matrix B");
    
    std::cout << "Vector v: ";
    for (std::size_t i = 0; i < 5; ++i) {
        std::cout << v(i) << " ";
    }
    std::cout << "\n\n";

    // ===== EXAMPLE 2: Matrix Multiplication =====
    std::cout << "EXAMPLE 2: Matrix multiplication using pool\n";
    std::cout << "--------------------------------------------\n";
    
    PoolMatrix3x3 C(alloc);
    
    // C = A * B
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            C(i, j) = 0;
            for (std::size_t k = 0; k < 3; ++k) {
                C(i, j) += A(i, k) * B(k, j);
            }
        }
    }
    
    print_matrix(C, "C = A * B");

    // ===== EXAMPLE 3: Simulating Inference Pipeline =====
    std::cout << "EXAMPLE 3: Simulating inference with layer tensors\n";
    std::cout << "---------------------------------------------------\n";
    
    // Create a new pool for inference
    std::size_t inference_pool_size = 10 * 1024; // 10KB
    MemoryBuffer inference_pool(inference_pool_size);
    auto inf_alloc = inference_pool.get_allocator<float>();
    
    // Simulate layer outputs
    using Layer1Out = PoolTensor<float, std::extents<std::size_t, 128>>;
    using Layer2Out = PoolTensor<float, std::extents<std::size_t, 64>>;
    using Layer3Out = PoolTensor<float, std::extents<std::size_t, 10>>;
    
    Layer1Out layer1_output(inf_alloc);
    Layer2Out layer2_output(inf_alloc);
    Layer3Out layer3_output(inf_alloc);
    
    // Initialize with dummy data
    for (std::size_t i = 0; i < 128; ++i) {
        layer1_output(i) = 0.1f * i;
    }
    
    // Simulate forward pass through layers
    std::cout << "Layer 1 output (first 5): ";
    for (std::size_t i = 0; i < 5; ++i) {
        std::cout << layer1_output(i) << " ";
    }
    std::cout << "\n";
    
    std::cout << "\nCreated layer outputs:\n";
    std::cout << "  Layer 1: 128 elements\n";
    std::cout << "  Layer 2: 64 elements\n";
    std::cout << "  Layer 3: 10 elements\n\n";

    // ===== EXAMPLE 4: Loading from Weights =====
    std::cout << "EXAMPLE 4: Loading tensor from pre-defined weights\n";
    std::cout << "---------------------------------------------------\n";
    
    // Simulate loading weights (e.g., from a model file)
    float weights[9] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
    };
    
    PoolMatrix3x3 identity(weights, alloc);
    print_matrix(identity, "Identity Matrix (from weights)");

    std::cout << "========================================\n";
    std::cout << "  Demo Complete!\n";
    std::cout << "========================================\n";
    std::cout << "\nKey takeaways:\n";
    std::cout << "1. MemoryBuffer provides fast bump allocation\n";
    std::cout << "2. Multiple tensors can share the same pool\n";
    std::cout << "3. No fragmentation, predictable memory usage\n";
    std::cout << "4. Perfect for inference where memory needs are known\n";

    return 0;
}

// clang++ -std=c++23 -O3 -Wall -Wextra testingout.cpp ../src/memorybuffer.cpp -o testingout && ./testingout