#ifndef __AVG_POOL_H__
#define __AVG_POOL_H__

#include <concepts>
#include <type_traits>
#include <utility>

#include "manual_blas.h"
#include "tensor.h"

template <size_t N, size_t output_size>
struct adaptive_avgpool1d_params {
    static_assert(output_size > 0, "AdaptiveAvgPool1DInPlace requires output_size > 0");
    static_assert(output_size <= N, "AdaptiveAvgPool1DInPlace requires output_size <= input size");

    static constexpr size_t stride = (output_size == 1) ? 1 : ((N - 1) / (output_size - 1));
    static constexpr size_t kernel_size = N - ((output_size - 1) * stride);

    static_assert(kernel_size > 0, "AdaptiveAvgPool1DInPlace computed invalid kernel size");
    static_assert(((N - kernel_size) / stride) + 1 == output_size,
        "AdaptiveAvgPool1DInPlace parameterization failed to match output_size");
};

template <size_t N, size_t K, size_t S, size_t OutIdx, class InT, class OutT, size_t... Is>
constexpr OutT avgpool1d_point_unrolled_impl(const InT* x_ptr, std::index_sequence<Is...>) {
    OutT acc{};
    ((acc += static_cast<OutT>(x_ptr[(OutIdx * S) + Is])), ...);
    return acc / static_cast<OutT>(K);
}

template <size_t N, size_t K, size_t S, size_t OutIdx, class InT, class OutT>
constexpr OutT avgpool1d_point_unrolled(const InT* x_ptr) {
    static_assert((OutIdx * S) + K <= N, "AdaptiveAvgPool1DInPlace unrolled access out of bounds");
    return avgpool1d_point_unrolled_impl<N, K, S, OutIdx, InT, OutT>(x_ptr, std::make_index_sequence<K>{});
}

template <size_t N, size_t K, size_t S, class InT, class OutT, size_t... Os>
constexpr void avgpool1d_full_unrolled(OutT* out_ptr, const InT* x_ptr, std::index_sequence<Os...>) {
    ((out_ptr[Os] = avgpool1d_point_unrolled<N, K, S, Os, InT, OutT>(x_ptr)), ...);
}

template <size_t output_size, class XTens, class OutTens>
requires (
    XTens::rank == 1 &&
    OutTens::rank == 1 &&
    std::same_as<std::remove_cv_t<typename XTens::value_type>, std::remove_cv_t<typename OutTens::value_type>> &&
    !std::is_const_v<typename OutTens::value_type>
)
void AdaptiveAvgPool1DInPlace(
    const XTens& x,
    OutTens& out
) {
    using in_t = typename XTens::value_type;
    using out_t = typename OutTens::value_type;
    using T = std::remove_cv_t<in_t>;
    static constexpr size_t N = XTens::extents_type::static_extent(0);
    using params = adaptive_avgpool1d_params<N, output_size>;
    static constexpr size_t stride = params::stride;
    static constexpr size_t kernel_size = params::kernel_size;
    static_assert(
        OutTens::extents_type::static_extent(0) == output_size,
        "AdaptiveAvgPool1DInPlace output tensor extent mismatch"
    );

    const in_t* x_ptr = x.data();
    out_t* out_ptr = out.data();

#if defined(__riscv_vector)
    if constexpr (output_size == 1 && std::same_as<T, float>) {
        out_ptr[0] = blas::sum_raw<T>(x_ptr, kernel_size) / static_cast<T>(kernel_size);
        return;
    }
#endif

    // Fixed-size tensors: fully unroll output points and window reductions for small workloads.
    if constexpr (output_size * kernel_size <= 1024) {
        avgpool1d_full_unrolled<N, kernel_size, stride, in_t, out_t>(
            out_ptr, x_ptr, std::make_index_sequence<output_size>{}
        );
        return;
    }

    for (size_t o = 0; o < output_size; ++o) {
        const size_t start = o * stride;
        out_t acc{};
        for (size_t k = 0; k < kernel_size; ++k) {
            acc += static_cast<out_t>(x_ptr[start + k]);
        }
        out_ptr[o] = acc / static_cast<out_t>(kernel_size);
    }
}



#endif
