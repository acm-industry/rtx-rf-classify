#ifndef __BATCH_NORM_H__
#define __BATCH_NORM_H__
#include <concepts>
#include <cmath>
#include <type_traits>

#include "ExprSystem/Broadcast.h"
#include "ExprSystem/Scalar.h"
#include "manual_blas.h"
#include "tensor.h"

// weights are 4, C as:
// [gamma, beta, mean, variance]
template <class XTens> requires ( XTens::rank == 2 )
void BatchNorm1DInPlace( 
    const XTens& x, 
    const TensorBase<typename XTens::value_type, std::extents<size_t, 4, XTens::static_extent(0) >>& weights, 
    TensorBase<typename XTens::value_type, typename XTens::extents_type>& out,
    float eps = 1e-5
) {
    auto gamma = weights[0];
    auto beta = weights[1];
    auto mean = weights[2];
    auto variance = weights[3];

    constexpr size_t C = XTens::static_extent(0);
    constexpr size_t L = XTens::static_extent(1);
    using T = std::remove_cv_t<typename XTens::value_type>;

    for (size_t j = 0; j < C; ++j) {
        auto gj = gamma[j];
        auto bj = beta[j];
        auto mj = mean[j];
        auto vj = variance[j];
        auto xj = x[j];
        auto outj = out[j];
        const T inv_std = static_cast<T>(1) / std::sqrt(vj + static_cast<T>(eps));
        const T scale = gj * inv_std;
        const T offset = bj - (gj * mj * inv_std);

        if constexpr (std::same_as<T, float>) {
#if defined(__riscv_vector)
            blas::affine_raw<float>(xj.data(), outj.data(), L, scale, offset);
            continue;
#endif
        }

        auto a = broadcast<std::extents<size_t, L>>(scale);
        auto b = broadcast<std::extents<size_t, L>>(offset);
        in_place_eval(a * xj + b, outj);
    }
}



#endif
