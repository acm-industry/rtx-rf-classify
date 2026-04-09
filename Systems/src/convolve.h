#ifndef __CONVOLVE_H__
#define __CONVOLVE_H__

#include <type_traits>
#include <utility>

#include "tensor.h"
#include "blas_ops.h"
#include "precision.h"

template <std::size_t XN, std::size_t KN, std::size_t Padding, std::size_t OutIdx, class T, std::size_t... Is>
constexpr T conv1d_point_unrolled_impl(const T* x_ptr, const T* w_ptr, std::index_sequence<Is...>) {
    T acc{};
    (
        [&] constexpr {
            constexpr std::size_t padded_idx = OutIdx + Is;
            if constexpr (padded_idx >= Padding && (padded_idx - Padding) < XN) {
                acc += x_ptr[padded_idx - Padding] * w_ptr[Is];
            }
        }(),
        ...
    );
    return acc;
}

template <std::size_t XN, std::size_t KN, std::size_t Padding, std::size_t OutIdx, class T>
constexpr T conv1d_point_unrolled(const T* x_ptr, const T* w_ptr) {
    return conv1d_point_unrolled_impl<XN, KN, Padding, OutIdx, T>(
        x_ptr, w_ptr, std::make_index_sequence<KN>{}
    );
}

template <std::size_t XN, std::size_t KN, std::size_t Padding, class T, std::size_t... Os>
constexpr void conv1d_full_unrolled(T* out_ptr, const T* x_ptr, const T* w_ptr, std::index_sequence<Os...>) {
    ((out_ptr[Os] = conv1d_point_unrolled<XN, KN, Padding, Os>(x_ptr, w_ptr)), ...);
}


template <
    std::size_t padding,
    class XTens,
    class WeightTens,
    Allocator<typename XTens::value_type> A = std::allocator<typename XTens::value_type>
>
requires (
    XTens::rank == 1 &&
    WeightTens::rank == 1 &&
    std::same_as<typename XTens::value_type, typename WeightTens::value_type> &&
    SupportedScalar<typename XTens::value_type>
)
auto Conv1D(const XTens& x, const WeightTens& weight, const A& allocator = A()) {
    using T = typename XTens::value_type;
    constexpr std::size_t XN = XTens::extents_type::static_extent(0);
    constexpr std::size_t KN = WeightTens::extents_type::static_extent(0);
    using kernel_extents = std::extents<std::size_t, KN>;
    using kernel_window_t = TensorBase<const T, kernel_extents>;

    static_assert(KN > 0, "Conv1D requires a non-empty kernel");
    static_assert(XN + (2 * padding) >= KN, "Conv1D output extent would be zero/negative");

    constexpr std::size_t ON = XN + (2 * padding) - KN + 1;
    using out_extents = std::extents<std::size_t, ON>;
    DynTensor<T, out_extents, A> out(allocator);

    const T* x_ptr = x.data();
    const T* w_ptr = weight.data();
    T* out_ptr = out.data();

    // For small/medium fixed tensors, fully unroll all output points and kernel taps.
    if constexpr (ON * KN <= 1024) {
        conv1d_full_unrolled<XN, KN, padding>(
            out_ptr, x_ptr, w_ptr, std::make_index_sequence<ON>{}
        );
        return out;
    }

    // Fallback: runtime outer loop with unrolled inner dot in the no-padding core region.
    constexpr bool has_core = (XN >= KN);
    constexpr std::size_t core_begin = has_core ? padding : 0;
    constexpr std::size_t core_end = has_core ? (padding + XN - KN + 1) : 0;

    for (std::size_t o = 0; o < ON; ++o) {
        if constexpr (padding == 0) {
            const kernel_window_t x_window{std::span<const T, KN>(x_ptr + o, KN)};
            out_ptr[o] = blas::dot(x_window.flat_view(), weight.flat_view());
            continue;
        }

        if (o >= core_begin && o < core_end) {
            const kernel_window_t x_window{std::span<const T, KN>(x_ptr + (o - padding), KN)};
            out_ptr[o] = blas::dot(x_window.flat_view(), weight.flat_view());
            continue;
        }

        accumulation_type_t<T> acc{};
        for (std::size_t k = 0; k < KN; ++k) {
            const std::size_t padded_idx = o + k;
            if (padded_idx < padding) continue;
            const std::size_t xi = padded_idx - padding;
            if (xi >= XN) continue;
            acc += promote_for_math(x_ptr[xi]) * promote_for_math(w_ptr[k]);
        }
        out_ptr[o] = cast_from_accum<T>(acc);
    }

    return out;
}

template <
    std::size_t padding,
    class XTens,
    class WeightTens,
    class OutTens
>
requires (
    XTens::rank == 1 &&
    WeightTens::rank == 1 &&
    OutTens::rank == 1 &&
    std::same_as<std::remove_cv_t<typename XTens::value_type>, std::remove_cv_t<typename WeightTens::value_type>> &&
    std::same_as<std::remove_cv_t<typename XTens::value_type>, std::remove_cv_t<typename OutTens::value_type>> &&
    !std::is_const_v<typename OutTens::value_type> &&
    (
        SupportedScalar<std::remove_cv_t<typename XTens::value_type>>
    )
)
constexpr void Conv1DInPlace(const XTens& x, const WeightTens& weight, OutTens& out) {
    using in_t = typename XTens::value_type;
    using out_t = typename OutTens::value_type;
    using T = std::remove_cv_t<in_t>;
    constexpr std::size_t XN = XTens::extents_type::static_extent(0);
    constexpr std::size_t KN = WeightTens::extents_type::static_extent(0);
    constexpr std::size_t ON = XN + (2 * padding) - KN + 1;
    using kernel_extents = std::extents<std::size_t, KN>;
    using kernel_window_t = TensorBase<const T, kernel_extents>;

    static_assert(KN > 0, "Conv1DInPlace requires a non-empty kernel");
    static_assert(XN + (2 * padding) >= KN, "Conv1DInPlace output extent would be zero/negative");
    static_assert(
        OutTens::extents_type::static_extent(0) == ON,
        "Conv1DInPlace output tensor extent mismatch"
    );

    const in_t* x_ptr = x.data();
    const in_t* w_ptr = weight.data();
    out_t* out_ptr = out.data();
    const kernel_window_t w_window{std::span<const T, KN>(w_ptr, KN)};

    if constexpr (ON * KN <= 1024) {
        conv1d_full_unrolled<XN, KN, padding>(
            out_ptr, x_ptr, w_ptr, std::make_index_sequence<ON>{}
        );
        return;
    }

    constexpr bool has_core = (XN >= KN);
    constexpr std::size_t core_begin = has_core ? padding : 0;
    constexpr std::size_t core_end = has_core ? (padding + XN - KN + 1) : 0;

    for (std::size_t o = 0; o < ON; ++o) {
        if constexpr (padding == 0) {
            const kernel_window_t x_window{std::span<const T, KN>(x_ptr + o, KN)};
            out_ptr[o] = blas::dot(x_window.flat_view(), w_window.flat_view());
            continue;
        }

        if (o >= core_begin && o < core_end) {
            const kernel_window_t x_window{std::span<const T, KN>(x_ptr + (o - padding), KN)};
            out_ptr[o] = blas::dot(x_window.flat_view(), w_window.flat_view());
            continue;
        }

        accumulation_type_t<T> acc{};
        for (std::size_t k = 0; k < KN; ++k) {
            const std::size_t padded_idx = o + k;
            if (padded_idx < padding) continue;
            const std::size_t xi = padded_idx - padding;
            if (xi >= XN) continue;
            acc += promote_for_math(x_ptr[xi]) * promote_for_math(w_ptr[k]);
        }
        out_ptr[o] = cast_from_accum<out_t>(acc);
    }
}



#endif
