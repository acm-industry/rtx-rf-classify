#pragma once

#include "layers.hpp"

namespace cnn {

// Example reusable model: Conv1D -> ReLU -> Flatten -> Linear
template <class Alloc>
struct ExampleCNN {
    static constexpr std::size_t kInChannels = 1;
    static constexpr std::size_t kOutChannels = 2;
    static constexpr std::size_t kInputLength = 16;
    static constexpr std::size_t kKernelSize = 3;
    static constexpr std::size_t kNumClasses = 4;

    using Conv = Conv1D<kInChannels, kOutChannels, kInputLength, kKernelSize, Alloc>;
    using Act = ReLU;
    using Flat = Flatten1D<kOutChannels, Conv::L_out, Alloc>;
    using FC = Linear<kOutChannels * Conv::L_out, kNumClasses, Alloc>;

    using InputTensor = typename Conv::InTensor;
    using OutputTensor = typename FC::OutTensor;

    Conv conv;
    Act relu;
    Flat flatten;
    FC fc;

    explicit ExampleCNN(Alloc alloc)
        : conv(alloc), relu(), flatten(alloc), fc(alloc) {}

    OutputTensor operator()(const InputTensor& in) const {
        auto x = conv(in);
        auto y = relu(x);
        auto z = flatten(y);
        return fc(z);
    }
};

}  // namespace cnn
