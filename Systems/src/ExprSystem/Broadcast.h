#ifndef __BROADCAST_H__
#define __BROADCAST_H__

#include <cstddef>
#include <experimental/mdspan>

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


#endif