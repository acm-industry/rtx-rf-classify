#ifndef __BROADCAST_H__
#define __BROADCAST_H__

#include <cstddef>
#include <experimental/mdspan>
#include <utility>
#include "Expression.h"
// TODO: Make broadcasting implicit?

template <class E, size_t I>
constexpr size_t static_extent_or_one() {
    if constexpr (I < E::rank()) return E::static_extent(I);
    return 1;
}

template <size_t A, size_t B> requires( A == B || A == 1 || B == 1 )
constexpr size_t broadcast_dim() {
    return (A > B) ? A : B;
}

template <class E1, class E2>
struct broadcast_two {
    static constexpr size_t R = (E1::rank() > E2::rank()) ? E1::rank() : E2::rank();

    template <size_t... I>
    static constexpr auto make(std::index_sequence<I...>) {
        return std::extents<
            size_t,
            broadcast_dim<
                static_extent_or_one<E1, R - 1 - I>(),
                static_extent_or_one<E2, R - 1 - I>()
            >()...
        >{};
    }

    using type = decltype(make(std::make_index_sequence<R>{}));
};

template <class...>
struct broadcast_many;

template <class E>
struct broadcast_many<E> {
    using type = E;
};

template <class E1, class E2, class... Rest>
struct broadcast_many<E1, E2, Rest...> {
    using type =
        typename broadcast_many<
            typename broadcast_two<E1, E2>::type,
            Rest...
        >::type;
};

template <class... Extents>
using broadcast_extents = typename broadcast_many<Extents...>::type;

template <class Extents>
constexpr size_t broadcast_total_size() {
    size_t size = 1;
    for (size_t i = 0; i < Extents::rank(); ++i) size *= Extents::static_extent(i);
    return size;
}

template <Expression Expr, class TargetExtents>
struct BroadcastExpr {
    static constexpr auto extents = TargetExtents{};
    using value_type = typename Expr::value_type;

    static constexpr size_t source_size = Expr::iter_size();
    static constexpr size_t target_size = broadcast_total_size<TargetExtents>();
    static_assert(
        source_size == 1 || source_size == target_size,
        "BroadcastExpr requires source iter_size() == 1 or equal to target size"
    );

    Expr expr;

    constexpr BroadcastExpr(Expr e) : expr(std::move(e)) {}

    static constexpr size_t iter_size() { return target_size; }

    constexpr value_type access(size_t idx) const {
        if constexpr (source_size == 1) return expr.access(0);
        return expr.access(idx);
    }
};

template <class TargetExtents, Expression Expr>
constexpr auto broadcast(Expr expr) {
    return BroadcastExpr<Expr, TargetExtents>{std::move(expr)};
}

#define BROADCAST(from, to) broadcast<decltype(to)::extents_type>((from));

#endif
