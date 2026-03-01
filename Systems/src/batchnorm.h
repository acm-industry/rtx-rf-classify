#ifndef __BATCH_NORM_H__
#define __BATCH_NORM_H__
#include "tensor.h"
#include "ExprSystem/Broadcast.h"
#include "ExprSystem/Expression.h"
#include "ExprSystem/Scalar.h"
#include "ExprSystem/ExprFunctions.h"

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

    #define BCAST(var) broadcast<std::extents<size_t, L>>((var))

    for (size_t j = 0; j < C; ++j) {
        auto gj = gamma[j];
        auto bj = beta[j];
        auto mj = mean[j];
        auto vj = variance[j];

        auto inv_std = 1.0f / std::sqrt( vj + eps );
        auto a = BCAST(gj * inv_std);
        auto b = BCAST(bj - gj * mj * inv_std);

        auto outj = out[j];
        in_place_eval( a * x[j] + b, outj );
    }

    #undef BCAST
}



#endif
