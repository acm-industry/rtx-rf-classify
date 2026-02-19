#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "layers.hpp"

namespace {

using Alloc = MemoryBuffer::Allocator<float>;
using Conv1 = cnn::Conv1D<1, 256, 130, 3, Alloc>;
using Conv2ByRow = cnn::Conv1D<256, 80, 132, 3, Alloc>;
using Flat = cnn::Flatten1D<80, 130, Alloc>;
using FC1 = cnn::Linear<10400, 256, Alloc>;
using FC2 = cnn::Linear<256, 11, Alloc>;

constexpr std::size_t kConv1W = 256 * 1 * 3;
constexpr std::size_t kConv1B = 256;
constexpr std::size_t kConv2W = 80 * 256 * 2 * 3;
constexpr std::size_t kConv2B = 80;
constexpr std::size_t kFC1W = 256 * 10400;
constexpr std::size_t kFC1B = 256;
constexpr std::size_t kFC2W = 11 * 256;
constexpr std::size_t kFC2B = 11;

struct Weights {
    std::vector<float> conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b;
};

template <typename T>
void read_vec(std::ifstream& ifs, std::vector<T>& out, std::size_t n) {
    out.resize(n);
    ifs.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(n * sizeof(T)));
    if (!ifs) throw std::runtime_error("Unexpected EOF in weights file");
}

Weights load_weights(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) throw std::runtime_error("Cannot open weights file: " + path);

    std::array<char, 4> magic{};
    ifs.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (magic != std::array<char, 4>{'R', 'M', 'L', 'W'}) throw std::runtime_error("Bad file magic");

    std::uint32_t version = 0;
    ifs.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != 1u) throw std::runtime_error("Unsupported weights version");

    Weights w;
    read_vec(ifs, w.conv1_w, kConv1W);
    read_vec(ifs, w.conv1_b, kConv1B);
    read_vec(ifs, w.conv2_w, kConv2W);
    read_vec(ifs, w.conv2_b, kConv2B);
    read_vec(ifs, w.fc1_w, kFC1W);
    read_vec(ifs, w.fc1_b, kFC1B);
    read_vec(ifs, w.fc2_w, kFC2W);
    read_vec(ifs, w.fc2_b, kFC2B);
    return w;
}

void load_model_weights(const Weights& w, Conv1& conv1, Conv2ByRow& conv2_r0, Conv2ByRow& conv2_r1, FC1& fc1,
                        FC2& fc2) {
    for (std::size_t o = 0; o < 256; ++o) {
        for (std::size_t k = 0; k < 3; ++k) {
            conv1.kernels.flat((o * 1 + 0) * 3 + k) = w.conv1_w[((o * 1 + 0) * 1 + 0) * 3 + k];
        }
        conv1.bias.flat(o) = w.conv1_b[o];
    }

    for (std::size_t o = 0; o < 80; ++o) {
        for (std::size_t i = 0; i < 256; ++i) {
            for (std::size_t k = 0; k < 3; ++k) {
                const std::size_t dst = (o * 256 + i) * 3 + k;
                conv2_r0.kernels.flat(dst) = w.conv2_w[((o * 256 + i) * 2 + 0) * 3 + k];
                conv2_r1.kernels.flat(dst) = w.conv2_w[((o * 256 + i) * 2 + 1) * 3 + k];
            }
        }
        conv2_r0.bias.flat(o) = 0.0f;
        conv2_r1.bias.flat(o) = 0.0f;
    }

    for (std::size_t i = 0; i < kFC1W; ++i) fc1.weights.flat(i) = w.fc1_w[i];
    for (std::size_t i = 0; i < kFC1B; ++i) fc1.bias.flat(i) = w.fc1_b[i];
    for (std::size_t i = 0; i < kFC2W; ++i) fc2.weights.flat(i) = w.fc2_w[i];
    for (std::size_t i = 0; i < kFC2B; ++i) fc2.bias.flat(i) = w.fc2_b[i];

    conv1.invalidate_fft_cache();
    conv2_r0.invalidate_fft_cache();
    conv2_r1.invalidate_fft_cache();
}

void pad_1x128_to_1x130(const TensorBase<float, std::extents<std::size_t, 1, 128>, Alloc>& in,
                        TensorBase<float, std::extents<std::size_t, 1, 130>, Alloc>& out) {
    for (std::size_t i = 0; i < out.total_size(); ++i) out.flat(i) = 0.0f;
    for (std::size_t x = 0; x < 128; ++x) out.flat(x + 1) = in.flat(x);
}

void pad_256x128_to_256x132(const TensorBase<float, std::extents<std::size_t, 256, 128>, Alloc>& in,
                            TensorBase<float, std::extents<std::size_t, 256, 132>, Alloc>& out) {
    for (std::size_t i = 0; i < out.total_size(); ++i) out.flat(i) = 0.0f;
    for (std::size_t c = 0; c < 256; ++c) {
        for (std::size_t x = 0; x < 128; ++x) {
            out.flat(c * 132 + (x + 2)) = in.flat(c * 128 + x);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::string weight_path = (argc > 1) ? std::string(argv[1]) : "radioml_cnn_weights.bin";

    try {
        auto weights = load_weights(weight_path);

        MemoryBuffer pool(1 << 25);
        Alloc alloc = pool.get_allocator<float>();

        Conv1 conv1(alloc);
        cnn::ReLU relu;
        Conv2ByRow conv2_r0(alloc);
        Conv2ByRow conv2_r1(alloc);
        Flat flatten(alloc);
        FC1 fc1(alloc);
        FC2 fc2(alloc);
        load_model_weights(weights, conv1, conv2_r0, conv2_r1, fc1, fc2);

        TensorBase<float, std::extents<std::size_t, 1, 2, 128>, Alloc> input(alloc);
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (std::size_t i = 0; i < input.total_size(); ++i) input.flat(i) = dist(rng);

        TensorBase<float, std::extents<std::size_t, 1, 128>, Alloc> row0(alloc), row1(alloc);
        for (std::size_t x = 0; x < 128; ++x) {
            row0.flat(x) = input.flat(x);
            row1.flat(x) = input.flat(128 + x);
        }

        TensorBase<float, std::extents<std::size_t, 1, 130>, Alloc> row0_p(alloc), row1_p(alloc);
        pad_1x128_to_1x130(row0, row0_p);
        pad_1x128_to_1x130(row1, row1_p);

        auto c1r0 = relu(conv1(row0_p));  // [256,128]
        auto c1r1 = relu(conv1(row1_p));  // [256,128]

        TensorBase<float, std::extents<std::size_t, 256, 132>, Alloc> c1r0_p(alloc), c1r1_p(alloc);
        pad_256x128_to_256x132(c1r0, c1r0_p);
        pad_256x128_to_256x132(c1r1, c1r1_p);

        auto c2r0 = conv2_r0(c1r0_p);  // [80,130]
        auto c2r1 = conv2_r1(c1r1_p);  // [80,130]

        TensorBase<float, std::extents<std::size_t, 80, 130>, Alloc> c2sum(alloc);
        for (std::size_t o = 0; o < 80; ++o) {
            for (std::size_t x = 0; x < 130; ++x) {
                const std::size_t idx = o * 130 + x;
                c2sum.flat(idx) = c2r0.flat(idx) + c2r1.flat(idx) + weights.conv2_b[o];
            }
        }

        auto a2 = relu(c2sum);
        auto flat = flatten(a2);
        auto h1 = relu(fc1(flat));
        auto logits = fc2(h1);

        std::size_t pred = 0;
        float best = logits.flat(0);
        for (std::size_t i = 1; i < logits.total_size(); ++i) {
            if (logits.flat(i) > best) {
                best = logits.flat(i);
                pred = i;
            }
        }

        std::cout << "input_shape=(1,1,2,128)\n";
        std::cout << "logits_shape=(1,11)\n";
        std::cout << "predicted_class=" << pred << "\n";
        std::cout << "first_sample_logits=[";
        for (std::size_t i = 0; i < logits.total_size(); ++i) {
            std::cout << logits.flat(i) << (i + 1 == logits.total_size() ? "" : ", ");
        }
        std::cout << "]\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
