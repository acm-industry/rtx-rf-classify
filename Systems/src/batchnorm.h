#ifndef __BATCH_NORM_H__
#define __BATCH_NORM_H__
#include "tensor.h"
#include "ExprSystem/Broadcast.h"
#include "ExprSystem/Expression.h"
#include "ExprSystem/Scalar.h"
#include "ExprSystem/ExprFunctions.h"
#include "precision.h"

// weights are 4, C as:
// [gamma, beta, mean, variance]
template <class XTens> requires ( XTens::rank == 2 )
void BatchNorm1DInPlace( 
    const XTens& x, 
    const TensorBase<typename XTens::value_type, std::extents<size_t, 4, XTens::static_extent(0) >>& weights, 
    TensorBase<typename XTens::value_type, typename XTens::extents_type>& out,
    float eps = 1e-5
) {
    using T = typename XTens::value_type;
    using accum_t = accumulation_type_t<T>;
    
    auto gamma = weights[0];
    auto beta = weights[1];
    auto mean = weights[2];
    auto variance = weights[3];

    constexpr size_t C = XTens::static_extent(0);
    constexpr size_t L = XTens::static_extent(1);

    #define BCAST(var) broadcast<std::extents<size_t, L>>((var))

    for (size_t j = 0; j < C; ++j) {
        accum_t gj = promote_for_math(gamma[j]);
        accum_t bj = promote_for_math(beta[j]);
        accum_t mj = promote_for_math(mean[j]);
        accum_t vj = promote_for_math(variance[j]);

        accum_t inv_std = accum_t{1} / static_cast<accum_t>(scalar_sqrt(vj + static_cast<accum_t>(eps)));
        auto a = BCAST(static_cast<T>(gj * inv_std));
        auto b = BCAST(static_cast<T>(bj - gj * mj * inv_std));

        auto outj = out[j];
        in_place_eval( a * x[j] + b, outj );
    }

    #undef BCAST
}



#endif
