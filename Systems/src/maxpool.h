#ifndef __MAX_POOL_H__
#define __MAX_POOL_H__

#include "tensor.h"
#include <array>
#include <cmath>
#include <type_traits>

template <size_t len, size_t k_len, size_t stride> requires ( len >= k_len )
constexpr size_t max_pool1d_length() {
    return static_cast<size_t>( static_cast<double>( len - k_len ) / stride ) + 1;
}

template <size_t kernel_size, size_t stride, class XTens, class OutTens>
requires(
    XTens::rank == 1 && OutTens::rank == 1 &&
    (XTens::extents_type::static_extent(0) > kernel_size) &&
    std::same_as<std::remove_cv_t<typename XTens::value_type>,
                 std::remove_cv_t<typename OutTens::value_type>> &&
    !std::is_const_v<typename OutTens::value_type> &&
    (OutTens::extents_type::static_extent(0) ==
     max_pool1d_length<XTens::extents_type::static_extent(0), kernel_size, stride>())
)
void MaxPool1DInPlace(const XTens& x, OutTens& out) {

    static constexpr size_t N = XTens::extents_type::static_extent(0);
    static constexpr size_t k = kernel_size;

    size_t out_idx = 0;

    std::array<size_t, k> dq;
    size_t head = 0, tail = 0, size = 0;

    for (size_t i = 0; i < N; ++i) {
        if ( size > 0 && dq[head] + k <= i ) {
            head = (head + 1) % k;
            --size;
        }

        while (size > 0) {
            size_t last = ( tail + k - 1 ) % k;
            if ( x[ dq[last] ] >= x[i] ) break;

            tail = last;
            --size;
        }

        dq[tail] = i;
        tail = ( tail + 1 ) % k;
        ++size;

        if ( i >= k - 1 ) {
            size_t win_start = i - (k - 1);
            if ( win_start % stride == 0 ) out[ out_idx++ ] = x[dq[head]];
        }


    }

}



#endif
