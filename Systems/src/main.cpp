#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <ranges>
#include <span>

#include "ExprSystem/Broadcast.h"
#include "ExprSystem/ExprFunctions.h"
#include "ExprSystem/Expression.h"
#include "ExprSystem/Scalar.h"
#include "avgpool.h"
#include "batchnorm.h"
#include "convolve.h"
#include "in_out.h"
#include "maweights.h"
#include "maxpool.h"
#include "memorybuffer.h"
#include "sequential.h"
#include "tensor.h"

static uint32_t from_big_endian(uint32_t value) {
    return ((value & 0x000000ffu) << 24) | ((value & 0x0000ff00u) << 8) |
           ((value & 0x00ff0000u) >> 8) | ((value & 0xff000000u) >> 24);
}

template <size_t... idxs>
    requires(sizeof...(idxs) > 0)
using f32Tensor = DynTensor<
    float, std::extents<size_t, idxs...>, MemoryBuffer::Allocator<float, 32>>;

template <size_t... idxs>
    requires(sizeof...(idxs) > 0)
using f32TensorView = TensorBase<float, std::extents<size_t, idxs...>>;

template <
    size_t padding, FixedExtent E1, FixedExtent E2, FixedExtent E3,
    FixedExtent E4>
    requires(
        E1::rank() == 2 && E2::rank() == 3 && E3::rank() == 1 && E4::rank() == 2
    )
void BroadcastConvolve(
    const TensorBase<float, E1>& input, const TensorBase<float, E2>& weights,
    const TensorBase<float, E3>& bias, TensorBase<float, E4>& out
) {
    std::array<float, E4::static_extent(1)> membuf;
    f32TensorView<E4::static_extent(1)> partial{
        std::span<float, E4::static_extent(1)>{membuf}
    };

    for (size_t oc = 0; oc < E2::static_extent(0); ++oc) {
        auto out_oc = out[oc];

        float b = bias[oc];
        for (size_t i = 0; i < E4::static_extent(1); ++i) out_oc[i] = b;

        for (size_t ic = 0; ic < E2::static_extent(1); ++ic) {
            auto x_ic = input[ic];
            auto w_oc_ic = weights[oc][ic];

            Conv1DInPlace<padding>(x_ic, w_oc_ic, partial);

            for (size_t i = 0; i < E4::static_extent(1); ++i)
                out_oc[i] += partial[i];
        }
    }
}

namespace model {
    using scratch_alloc_t = MemoryBuffer::Allocator<std::byte, 32UL>;

    using In = f32TensorView<3, 128>;
    using L1Conv = f32TensorView<64, 128>;
    using L1Pool = f32TensorView<64, 64>;
    using L2Conv = f32TensorView<128, 64>;
    using L2Pool = f32TensorView<128, 32>;
    using L3Conv = f32TensorView<128, 32>;
    using L4Conv = f32TensorView<128, 32>;
    using Vec128 = f32TensorView<128>;
    using Vec11 = f32TensorView<11>;

    template <
        size_t padding, class XIn, class XOut, size_t OC, size_t IC, size_t K>
    struct BroadcastConv1D {
        f32TensorView<OC, IC, K> w;
        f32TensorView<OC> b;

        BroadcastConv1D(float* weights, float* bias)
            : w(std::span<float, OC * IC * K>(weights, OC * IC * K)),
              b(std::span<float, OC>(bias, OC)) {}

        void operator()(const XIn& in, XOut& out) const {
            BroadcastConvolve<padding>(in, w, b, out);
        }
    };

    template <class X, size_t C>
    struct BatchNorm {
        f32TensorView<4, C> w;

        explicit BatchNorm(float* weights)
            : w(std::span<float, 4 * C>(weights, 4 * C)) {}

        void operator()(const X& in, X& out) const {
            BatchNorm1DInPlace(in, w, out);
        }
    };

    template <class X>
    struct ReluInPlace {
        void operator()(const X& in, X& out) const {
            in_place_eval(relu(in), out);
        }
    };

    template <size_t kernel, size_t stride, class XIn, class XOut>
        requires(
            XIn::rank == 2 && XOut::rank == 2 &&
            std::same_as<
                std::remove_cv_t<typename XIn::value_type>,
                std::remove_cv_t<typename XOut::value_type>> &&
            !std::is_const_v<typename XOut::value_type> &&
            (XIn::extents_type::static_extent(0) ==
             XOut::extents_type::static_extent(0))
        )
    struct ChannelMaxPool1D {
        static constexpr size_t C = XIn::extents_type::static_extent(0);

        void operator()(const XIn& in, XOut& out) const {
            for (size_t i = 0; i < C; ++i) {
                auto outvec = out[i];
                MaxPool1DInPlace<kernel, stride>(in[i], outvec);
            }
        }
    };

    using Conv1 = BroadcastConv1D<2, In, L1Conv, 64, 3, 5>;
    using Conv2 = BroadcastConv1D<2, L1Pool, L2Conv, 128, 64, 5>;
    using Conv3 = BroadcastConv1D<1, L2Pool, L3Conv, 128, 128, 3>;
    using Conv4 = BroadcastConv1D<1, L3Conv, L4Conv, 128, 128, 3>;

    using BN1 = BatchNorm<L1Conv, 64>;
    using BN2 = BatchNorm<L2Conv, 128>;
    using BN3 = BatchNorm<L3Conv, 128>;
    using BN4 = BatchNorm<L4Conv, 128>;

    using Relu64_128 = ReluInPlace<L1Conv>;
    using Relu128_64 = ReluInPlace<L2Conv>;
    using Relu128_32 = ReluInPlace<L3Conv>;
    using ReluOut = ReluInPlace<L4Conv>;

    using MP1 = ChannelMaxPool1D<2, 2, L1Conv, L1Pool>;
    using MP2 = ChannelMaxPool1D<2, 2, L2Conv, L2Pool>;

    struct GlobalAvgPool {
        void operator()(const L4Conv& in, Vec128& out) const {
            for (size_t i = 0; i < 128; ++i) {
                auto in_slice = in[i];
                TensorBase<float, std::extents<size_t, 1>> out_scalar{
                    std::span<float, 1>(out.data() + i, 1)
                };
                AdaptiveAvgPool1DInPlace<1>(in_slice, out_scalar);
            }
        }
    };

    template <class XIn, class XOut, size_t O, size_t I>
    struct Linear {
        f32TensorView<O, I> w;

        explicit Linear(float* weights)
            : w(std::span<float, O * I>(weights, O * I)) {}

        void operator()(const XIn& in, XOut& out) const {
            blas::gemv(w, in, out);
        }
    };

    template <class X, size_t N>
    struct BiasAdd {
        X b;

        explicit BiasAdd(float* bias) : b(std::span<float, N>(bias, N)) {}

        void operator()(const X& in, X& out) const {
            in_place_eval(in + b, out);
        }
    };

    template <class X, size_t N>
    struct BiasAddRelu {
        X b;

        explicit BiasAddRelu(float* bias) : b(std::span<float, N>(bias, N)) {}

        void operator()(const X& in, X& out) const {
            in_place_eval(relu(in + b), out);
        }
    };

    using Linear1 = Linear<Vec128, Vec128, 128, 128>;
    using BiasRelu1 = BiasAddRelu<Vec128, 128>;
    using Linear2 = Linear<Vec128, Vec11, 11, 128>;
    using Bias2 = BiasAdd<Vec11, 11>;

    using FeatureExtractor = Sequential<
        scratch_alloc_t, Conv1, BN1, Relu64_128, MP1, Conv2, BN2, Relu128_64,
        MP2, Conv3, BN3, Relu128_32, Conv4, BN4, ReluOut>;

    using ClassifierHead = Sequential<
        scratch_alloc_t, GlobalAvgPool, Linear1, BiasRelu1, Linear2, Bias2>;

    using Model = Sequential<scratch_alloc_t, FeatureExtractor, ClassifierHead>;
}  // namespace model

int main() {
    static constexpr size_t INPUT_ALLOC_BYTES = 1024 * 1024;
    std::array<std::byte, INPUT_ALLOC_BYTES> memory_store;
    MemoryBuffer buf{std::span<std::byte, INPUT_ALLOC_BYTES>{memory_store}};

    alignas(32) std::array<float, 3 * 128> input_mem;
    f32TensorView<3, 128> input{std::span<float, 3 * 128>{input_mem}};

    model::scratch_alloc_t seq_alloc = buf.get_allocator<std::byte, 32>();
    model::FeatureExtractor feature_extractor{
        seq_alloc,
        model::Conv1{weights_conv_1d_l1, weights_conv_1d_bias_l1},
        model::BN1{weights_bn_1d_l1},
        model::Relu64_128{},
        model::MP1{},
        model::Conv2{weights_conv_1d_l2, weights_conv_1d_bias_l2},
        model::BN2{weights_bn_1d_l2},
        model::Relu128_64{},
        model::MP2{},
        model::Conv3{weights_conv_1d_l3, weights_conv_1d_bias_l3},
        model::BN3{weights_bn_1d_l3},
        model::Relu128_32{},
        model::Conv4{weights_conv_1d_l4, weights_conv_1d_bias_l4},
        model::BN4{weights_bn_1d_l4},
        model::ReluOut{}
    };
    model::ClassifierHead classifier_head{
        seq_alloc,
        model::GlobalAvgPool{},
        model::Linear1{linear_mat_1},
        model::BiasRelu1{linear_add_1},
        model::Linear2{linear_mat_2},
        model::Bias2{linear_add_2}
    };
    model::Model model{seq_alloc, feature_extractor, classifier_head};

    RFIO rfio;

    for (;;) {
        // protocol: 4 bytes of batch len, then batch_len instances of 3
        // * 128 bytes.

        uint32_t batch_len_wire;
        rfio.recv(
            std::span<std::byte>(
                reinterpret_cast<std::byte*>(&batch_len_wire),
                sizeof(batch_len_wire)
            )
        );
        uint32_t batch_len = from_big_endian(batch_len_wire);

        if (batch_len == 0) continue;

        for (uint32_t i = 0; i < batch_len; ++i) {
            rfio.recv(
                std::span<std::byte>(
                    reinterpret_cast<std::byte*>(input.data()),
                    3 * 128 * sizeof(float)
                )
            );

            std::array<float, 11> final_buf;

            f32TensorView<11> final{std::span<float, 11>{final_buf}};

            buf.reset();

            model(input, final);

            unsigned char argmax = 0;
            float best = final[0];
            for (unsigned char j = 1; j < 11; ++j) {
                float v = final[j];
                if (v > best) {
                    best = v;
                    argmax = j;
                }
            }

            rfio.send(
                std::span<std::byte>(
                    reinterpret_cast<std::byte*>(&argmax), 1
                )
            );
        }
    }

    return 0;
}
