#ifndef __EXPRESSION_H__
#define __EXPRESSION_H__

#include <cstddef>
#include <concepts>
#include <tuple>
#include "Broadcast.h"

template <class E>
concept Expression = requires(E a, size_t i) {
    typename E::reference; // Doesn't have to be a reference; can be a proxy object
    typename E::extents_type;

    { E::extents() } -> std::same_as<const typename E::extents_type&>;
    { E::iter_size() } -> std::convertible_to<size_t>; 
    { a.access(i) } -> std::same_as<typename E::reference>;
};




#endif