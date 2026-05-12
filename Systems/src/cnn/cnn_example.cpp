#include <iostream>

#include "cnn_example.hpp"

int main() {
    using Alloc = MemoryBuffer::Allocator<float>;

    MemoryBuffer pool(1 << 16);
    Alloc alloc = pool.get_allocator<float>();

    cnn::ExampleCNN<Alloc> model(alloc);

    typename cnn::ExampleCNN<Alloc>::InputTensor input(alloc);
    for (std::size_t i = 0; i < input.total_size(); ++i) {
        input.flat(i) = static_cast<float>(i) * 0.05f;
    }

    auto logits = model(input);

    std::cout << "FFTW backend enabled: " << (cnn::fftw_backend_available() ? "yes" : "no") << "\n";
    std::cout << "Output logits: ";
    for (std::size_t i = 0; i < logits.total_size(); ++i) {
        std::cout << logits.flat(i) << (i + 1 == logits.total_size() ? '\n' : ' ');
    }

    return 0;
}
