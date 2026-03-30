#ifndef SYSTEMS_OPERATORS_UNARY_OPS_H
#define SYSTEMS_OPERATORS_UNARY_OPS_H

#include <concepts>
#include <type_traits>

#include "../tensor.h"
#include "../vector/vector_ops.h"

namespace ops {

template <class InT, class OutT>
concept UnaryTensorPair =
    requires(const InT& in, OutT& out) {
        typename InT::value_type;
        typename OutT::value_type;
        { InT::static_size } -> std::convertible_to<std::size_t>;
        { OutT::static_size } -> std::convertible_to<std::size_t>;
        { in.flat_view().data() } -> std::convertible_to<const typename InT::value_type*>;
        { out.flat_view().data() } -> std::convertible_to<typename OutT::value_type*>;
    };

template <class InT, class OutT>
constexpr void unary_tensor_static_checks() {
    static_assert(UnaryTensorPair<InT, OutT>, "Unary op requires Tensor-like inputs with flat_view().");
    static_assert(InT::static_size == OutT::static_size, "Unary op requires equal input/output static_size.");
    static_assert(std::same_as<std::remove_cv_t<typename InT::value_type>, float>,
                  "Current vec backend supports float unary ops only.");
    static_assert(std::same_as<std::remove_cv_t<typename OutT::value_type>, float>,
                  "Current vec backend supports float unary ops only.");
}

struct ReluOp {
    template <class InT, class OutT>
    void operator()(const InT& in, OutT& out) const {
        unary_tensor_static_checks<InT, OutT>();
        vec::relu(in.flat_view().data(), out.flat_view().data(), InT::static_size);
    }
};

struct ExpOp {
    template <class InT, class OutT>
    void operator()(const InT& in, OutT& out) const {
        unary_tensor_static_checks<InT, OutT>();
        vec::exp(in.flat_view().data(), out.flat_view().data(), InT::static_size);
    }
};

struct LogOp {
    template <class InT, class OutT>
    void operator()(const InT& in, OutT& out) const {
        unary_tensor_static_checks<InT, OutT>();
        vec::log(in.flat_view().data(), out.flat_view().data(), InT::static_size);
    }
};

struct TanhOp {
    template <class InT, class OutT>
    void operator()(const InT& in, OutT& out) const {
        unary_tensor_static_checks<InT, OutT>();
        vec::tanh(in.flat_view().data(), out.flat_view().data(), InT::static_size);
    }
};

struct SigmoidOp {
    template <class InT, class OutT>
    void operator()(const InT& in, OutT& out) const {
        unary_tensor_static_checks<InT, OutT>();
        vec::sigmoid(in.flat_view().data(), out.flat_view().data(), InT::static_size);
    }
};

struct AffineOp {
    float alpha;
    float beta;

    explicit AffineOp(float a, float b) : alpha(a), beta(b) {}

    template <class InT, class OutT>
    void operator()(const InT& in, OutT& out) const {
        unary_tensor_static_checks<InT, OutT>();
        const auto in_flat = in.flat_view();
        auto out_flat = out.flat_view();
        for (std::size_t i = 0; i < InT::static_size; ++i) {
            out_flat[i] = alpha * in_flat[i] + beta;
        }
    }
};

}  // namespace ops

#endif
