#include <iostream>

#include "layers.hpp"

int main() {
    using Alloc = MemoryBuffer::Allocator<float>;
    MemoryBuffer pool(1 << 15);
    Alloc alloc = pool.get_allocator<float>();

    using Conv = cnn::Conv1D<1, 1, 8, 3, Alloc>;
    using Flat = cnn::Flatten1D<1, Conv::L_out, Alloc>;
    using FC = cnn::Linear<Conv::L_out, 2, Alloc>;

    Conv conv(alloc);
    cnn::ReLU relu;
    Flat flatten(alloc);
    FC fc(alloc);

    cnn::Sequential<Conv, cnn::ReLU, Flat, FC> net(conv, relu, flatten, fc);

    Conv::InTensor in(alloc);
    for (std::size_t i = 0; i < in.total_size(); ++i) {
        in.flat(i) = static_cast<float>(i + 1);
    }

    auto out = net(in);

    std::cout << "Sequential output: ";
    for (std::size_t i = 0; i < out.total_size(); ++i) {
        std::cout << out.flat(i) << (i + 1 == out.total_size() ? '\n' : ' ');
    }

    return 0;
}
