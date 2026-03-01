#ifndef __BLAS_OPS_H__
#define __BLAS_OPS_H__

#include <experimental/mdspan>
#include <concepts>
#include <type_traits>
#include <cblas.h>

/*
    Purpose: Type-safe, mdspan-aware BLAS wrappers with compile-time dimension checks.

    Provides gemm, gemv, and dot operations that accept std::mdspan with fixed extents,
    template on the floating-point type (float/double), and dispatch to the appropriate
    CBLAS routine. Dimension compatibility is enforced via static_assert.

    Usage: Include this header and link against OpenBLAS (or any CBLAS provider).
    Callers pass mdspan views directly - for TensorBase, use tensor.view().

    All operations assume row-major layout (CblasRowMajor), matching TensorBase's
    default std::layout_right.
*/

namespace blas {

template<typename M>
concept Matrix2D = requires {
    typename M::element_type;
    requires std::floating_point<typename M::element_type>;
    requires M::rank == 2;
    requires M::static_extent(0) != std::dynamic_extent;
    requires M::static_extent(1) != std::dynamic_extent;
};

template<typename V>
concept Vector1D = requires {
    typename V::element_type;
    requires std::floating_point<typename V::element_type>;
    requires V::rank == 1;
    requires V::static_extent(0) != std::dynamic_extent;
};

// C = A * B
// A is M x K, B is K x N, C is M x N (all row-major)
template<Matrix2D MA, Matrix2D MB, Matrix2D MC>
    requires std::same_as<typename MA::element_type, typename MB::element_type> &&
             std::same_as<typename MA::element_type, typename MC::element_type>
void gemm(const MA& a, const MB& b, MC& c) {
    using T = std::remove_cv_t<typename MA::element_type>;

    static constexpr auto M = MA::static_extent(0);
    static constexpr auto K = MA::static_extent(1);
    static constexpr auto N = MB::static_extent(1);

    static_assert(MB::static_extent(0) == K, "gemm: A columns must equal B rows");
    static_assert(MC::static_extent(0) == M, "gemm: C rows must equal A rows");
    static_assert(MC::static_extent(1) == N, "gemm: C columns must equal B columns");

    if constexpr (std::is_same_v<T, float>) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                    1.0f,
                    a.data(), static_cast<int>(K),
                    b.data(), static_cast<int>(N),
                    0.0f,
                    c.data(), static_cast<int>(N));
    } else if constexpr (std::is_same_v<T, double>) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                    1.0,
                    a.data(), static_cast<int>(K),
                    b.data(), static_cast<int>(N),
                    0.0,
                    c.data(), static_cast<int>(N));
    } else {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
            "gemm: only float and double are supported by CBLAS");
    }
}

// y = A * x
// A is M x N, x is N, y is M (row-major)
template<Matrix2D MA, Vector1D VX, Vector1D VY>
    requires std::convertible_to<typename MA::element_type, typename VX::element_type> &&
             std::convertible_to<typename MA::element_type, typename VY::element_type>
void gemv(const MA& a, const VX& x, VY& y) {
    using T = std::remove_cv_t<typename MA::element_type>;

    static constexpr auto M = MA::static_extent(0);
    static constexpr auto N = MA::static_extent(1);

    static_assert(VX::static_extent(0) == N, "gemv: x length must equal A columns");
    static_assert(VY::static_extent(0) == M, "gemv: y length must equal A rows");

    if constexpr (std::is_same_v<T, float>) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    static_cast<int>(M), static_cast<int>(N),
                    1.0f,
                    a.data(), static_cast<int>(N),
                    x.data(), 1,
                    0.0f,
                    y.data(), 1);
    } else if constexpr (std::is_same_v<T, double>) {
        cblas_dgemv(CblasRowMajor, CblasNoTrans,
                    static_cast<int>(M), static_cast<int>(N),
                    1.0,
                    a.data(), static_cast<int>(N),
                    x.data(), 1,
                    0.0,
                    y.data(), 1);
    } else {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
            "gemv: only float and double are supported by CBLAS");
    }
}

// result = a . b
// a and b are both length N
template<Vector1D VA, Vector1D VB>
    requires std::same_as<typename VA::element_type, typename VB::element_type>
auto dot(const VA& a, const VB& b) -> typename VA::element_type {
    using T = std::remove_cv_t<typename VA::element_type>;

    static constexpr auto N = VA::static_extent(0);
    static_assert(VB::static_extent(0) == N, "dot: vector lengths must match");

    if constexpr (std::is_same_v<T, float>) {
        return cblas_sdot(static_cast<int>(N), a.data(), 1, b.data(), 1);
    } else if constexpr (std::is_same_v<T, double>) {
        return cblas_ddot(static_cast<int>(N), a.data(), 1, b.data(), 1);
    } else {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
            "dot: only float and double are supported by CBLAS");
    }
}

} // namespace blas

#endif
