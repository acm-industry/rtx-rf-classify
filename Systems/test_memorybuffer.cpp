#include <iostream>
#include <list>
#include <string>
#include <vector>

#include "memorybuffer.h"

void test_basic() {
    std::cout << "Test 1: Basic allocation... ";
    MemoryBuffer buf(1024);
    auto alloc = buf.get_allocator<int>();

    int* p1 = alloc.allocate(5);
    int* p2 = alloc.allocate(3);

    for (int i = 0; i < 5; i++) p1[i] = i;
    for (int i = 0; i < 3; i++) p2[i] = i * 10;

    if (p2 > p1 && p1[4] == 4 && p2[2] == 20)
        std::cout << "PASSED\n";
    else
        std::cout << "FAILED\n";
}

void test_external_buffer() {
    std::cout << "Test 2: External buffer... ";
    std::byte external[512];
    MemoryBuffer buf(std::span<std::byte>(external, 512));

    double* p = buf.get_allocator<double>().allocate(10);
    p[0] = 3.14;

    if (external <= reinterpret_cast<std::byte*>(p) &&
        reinterpret_cast<std::byte*>(p) < external + 512)
        std::cout << "PASSED\n";
    else
        std::cout << "FAILED\n";
}

void test_vector() {
    std::cout << "Test 3: STL vector with custom allocator... ";
    MemoryBuffer buf(2048);
    auto alloc = buf.get_allocator<int>();

    std::vector<int, decltype(alloc)> vec(alloc);
    for (int i = 0; i < 10; i++) vec.push_back(i * i);

    if (vec.size() == 10 && vec[5] == 25)
        std::cout << "PASSED\n";
    else
        std::cout << "FAILED\n";
}

void test_overflow() {
    std::cout << "Test 4: Capacity overflow detection... ";
    MemoryBuffer buf(64);
    auto alloc = buf.get_allocator<int>();

    try {
        int* p = alloc.allocate(1000);
        (void)p;  // suppress unused variable warning
        std::cout << "FAILED (no exception thrown)\n";
    } catch (const std::bad_alloc&) {
        std::cout << "PASSED\n";
    }
}

void test_multiple_allocators() {
    std::cout << "Test 5: Multiple allocator instances... ";
    MemoryBuffer buf(1024);

    auto alloc1 = buf.get_allocator<int>();
    auto alloc2 = buf.get_allocator<double>();
    auto alloc3 = alloc1;

    int* p1 = alloc1.allocate(5);
    double* p2 = alloc2.allocate(3);
    int* p3 = alloc3.allocate(2);

    // All should come from same buffer sequentially
    if (reinterpret_cast<std::byte*>(p2) > reinterpret_cast<std::byte*>(p1) &&
        reinterpret_cast<std::byte*>(p3) > reinterpret_cast<std::byte*>(p2))
        std::cout << "PASSED\n";
    else
        std::cout << "FAILED\n";
}

void test_rebind() {
    std::cout << "Test 6: Rebind with std::list... ";
    MemoryBuffer buf(4096);
    auto alloc = buf.get_allocator<int>();

    // std::list uses rebind internally to allocate list nodes
    std::list<int, decltype(alloc)> lst(alloc);
    for (int i = 0; i < 10; i++) lst.push_back(i);

    int sum = 0;
    for (int val : lst) sum += val;

    if (lst.size() == 10 && sum == 45)
        std::cout << "PASSED\n";
    else
        std::cout << "FAILED\n";
}

void test_propagate_copy() {
    std::cout << "Test 7: Propagate on copy assignment... ";
    MemoryBuffer buf(1024);
    auto alloc1 = buf.get_allocator<int>();

    std::vector<int, decltype(alloc1)> vec1(alloc1);
    for (int i = 0; i < 5; i++) vec1.push_back(i);

    auto alloc2 = buf.get_allocator<int>();
    std::vector<int, decltype(alloc2)> vec2(alloc2);
    vec2 = vec1;

    if (vec2.size() == 5 && vec2[4] == 4 && vec2.get_allocator() == alloc1)
        std::cout << "PASSED\n";
    else
        std::cout << "FAILED\n";
}

void test_propagate_move() {
    std::cout << "Test 8: Propagate on move assignment... ";
    MemoryBuffer buf(1024);
    auto alloc1 = buf.get_allocator<int>();

    std::vector<int, decltype(alloc1)> vec1(alloc1);
    for (int i = 0; i < 5; i++) vec1.push_back(i);

    auto alloc2 = buf.get_allocator<int>();
    std::vector<int, decltype(alloc2)> vec2(alloc2);
    vec2 = std::move(vec1);  // move assignment

    if (vec2.size() == 5 && vec2[4] == 4 && vec2.get_allocator() == alloc1)
        std::cout << "PASSED\n";
    else
        std::cout << "FAILED\n";
}

void test_propagate_swap() {
    std::cout << "Test 9: Propagate on swap... ";
    MemoryBuffer buf(1024);
    auto alloc1 = buf.get_allocator<int>();

    std::vector<int, decltype(alloc1)> vec1(alloc1);
    for (int i = 0; i < 5; i++) vec1.push_back(i);

    auto alloc2 = buf.get_allocator<int>();
    std::vector<int, decltype(alloc2)> vec2(alloc2);
    for (int i = 10; i < 15; i++) vec2.push_back(i);

    vec1.swap(vec2);

    if (vec1.size() == 5 && vec1[0] == 10 && vec2.size() == 5 && vec2[0] == 0 &&
        vec1.get_allocator() == alloc2 && vec2.get_allocator() == alloc1)
        std::cout << "PASSED\n";
    else
        std::cout << "FAILED\n";
}

int main() {
    std::cout << "=== MemoryBuffer Test Suite ===\n\n";

    test_basic();
    test_external_buffer();
    test_vector();
    test_overflow();
    test_multiple_allocators();
    test_rebind();
    test_propagate_copy();
    test_propagate_move();
    test_propagate_swap();

    std::cout << "\n=== All tests complete ===\n";
    return 0;
}
