#ifndef __MATRIX_EVAL_H__
#define __MATRIX_EVAL_H__

#include "tensor.h"
#include "blas_ops.h"
#include "ExprSystem/Expression.h"

/*
    Purpose: Bridge between the Expression system (element-wise 1D eval) and
    BLAS matrix operations (2D). Materializes lazy expressions into TensorBase
    storage, then calls blas::gemm on the results.

    Usage:
        TensorBase<float, std::extents<size_t, 2, 3>> a;
        TensorBase<float, std::extents<size_t, 3, 2>> b;
        TensorBase<float, std::extents<size_t, 2, 2>> c;
        auto expr_a = sigmoid(some_expr_over_a);
        auto expr_b = some_expr_over_b;
        matrix_eval(expr_a, a, expr_b, b, c);
        // a and b now hold materialized expression results, c = a * b
*/

// Materialize an expression into a TensorBase via flat indexing,
// then perform gemm: out = mat_a * mat_b
template<std::floating_point T,
         Expression ExprA, FixedExtent EA, Allocator<T> AllocA,
         Expression ExprB, FixedExtent EB, Allocator<T> AllocB,
         FixedExtent EC, Allocator<T> AllocC>
    requires (EA::rank() == 2 && EB::rank() == 2 && EC::rank() == 2)
void matrix_eval(const ExprA& expr_a, TensorBase<T, EA, AllocA>& mat_a,
                 const ExprB& expr_b, TensorBase<T, EB, AllocB>& mat_b,
                 TensorBase<T, EC, AllocC>& out) {
    static_assert(ExprA::iter_size() == compute_static_size<EA>(),
        "matrix_eval: expression A size must match tensor A size");
    static_assert(ExprB::iter_size() == compute_static_size<EB>(),
        "matrix_eval: expression B size must match tensor B size");

    // Materialize expressions into tensor storage
    for (size_t i = 0; i < ExprA::iter_size(); ++i)
        mat_a.flat(i) = expr_a.access(i);

    for (size_t i = 0; i < ExprB::iter_size(); ++i)
        mat_b.flat(i) = expr_b.access(i);

    // Perform BLAS gemm on the materialized tensors
    blas::gemm(mat_a.view(), mat_b.view(), out.view());
}

// Overload when expressions are already materialized (no-op eval step).
// Directly calls blas::gemm on existing tensors.
template<std::floating_point T,
         FixedExtent EA, Allocator<T> AllocA,
         FixedExtent EB, Allocator<T> AllocB,
         FixedExtent EC, Allocator<T> AllocC>
    requires (EA::rank() == 2 && EB::rank() == 2 && EC::rank() == 2)
void matrix_eval(const TensorBase<T, EA, AllocA>& mat_a,
                 const TensorBase<T, EB, AllocB>& mat_b,
                 TensorBase<T, EC, AllocC>& out) {
    blas::gemm(mat_a.view(), mat_b.view(), out.view());
}

#endif
