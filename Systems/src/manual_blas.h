#ifndef __MANUAL_BLAS_H__
#define __MANUAL_BLAS_H__

#include <cstddef>
#include <concepts>
#include <type_traits>

namespace blas {

template <class V>
concept Vector1D =
    requires(const V& v) {
        typename V::element_type;
        { v.data() };
    } &&
    (V::rank == 1) &&
    std::floating_point<std::remove_cv_t<typename V::element_type>>;

template <class M>
concept Matrix2D =
    requires(const M& m) {
        typename M::element_type;
        { m.data() };
    } &&
    (M::rank == 2) &&
    std::floating_point<std::remove_cv_t<typename M::element_type>>;

template <class T>
T dot_raw(const T* a, const T* b, std::size_t n) {
#if defined(__riscv_vector)
    if constexpr (std::is_same_v<std::remove_cv_t<T>, float>) {
        const float zero = 0.0f;
        float sum = 0.0f;
        float chunk = 0.0f;
        auto* pa = reinterpret_cast<const float*>(a);
        auto* pb = reinterpret_cast<const float*>(b);
        std::size_t remaining = n;
        std::size_t vl = 0;
        std::size_t bytes = 0;

        asm volatile(
            "1:\n\t"
            "beqz %[remaining], 2f\n\t"
            "vsetvli %[vl], %[remaining], e32, m1, ta, ma\n\t"
            "vle32.v v0, (%[a])\n\t"
            "vle32.v v1, (%[b])\n\t"
            "vfmul.vv v2, v0, v1\n\t"
            "vfmv.s.f v3, %[zero]\n\t"
            "vfredusum.vs v3, v2, v3\n\t"
            "vfmv.f.s %[chunk], v3\n\t"
            "fadd.s %[sum], %[sum], %[chunk]\n\t"
            "slli %[bytes], %[vl], 2\n\t"
            "add %[a], %[a], %[bytes]\n\t"
            "add %[b], %[b], %[bytes]\n\t"
            "sub %[remaining], %[remaining], %[vl]\n\t"
            "j 1b\n\t"
            "2:\n\t"
            : [sum] "+f"(sum),
              [chunk] "=&f"(chunk),
              [a] "+r"(pa),
              [b] "+r"(pb),
              [remaining] "+r"(remaining),
              [vl] "=&r"(vl),
              [bytes] "=&r"(bytes)
            : [zero] "f"(zero)
            : "memory"
        );

        return sum;
    }
#endif

    T sum{};
    for (std::size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

template <class T>
T sum_raw(const T* x, std::size_t n) {
#if defined(__riscv_vector)
    if constexpr (std::is_same_v<std::remove_cv_t<T>, float>) {
        const float zero = 0.0f;
        float sum = 0.0f;
        float chunk = 0.0f;
        auto* px = reinterpret_cast<const float*>(x);
        std::size_t remaining = n;
        std::size_t vl = 0;
        std::size_t bytes = 0;

        asm volatile(
            "1:\n\t"
            "beqz %[remaining], 2f\n\t"
            "vsetvli %[vl], %[remaining], e32, m1, ta, ma\n\t"
            "vle32.v v0, (%[x])\n\t"
            "vfmv.s.f v1, %[zero]\n\t"
            "vfredusum.vs v1, v0, v1\n\t"
            "vfmv.f.s %[chunk], v1\n\t"
            "fadd.s %[sum], %[sum], %[chunk]\n\t"
            "slli %[bytes], %[vl], 2\n\t"
            "add %[x], %[x], %[bytes]\n\t"
            "sub %[remaining], %[remaining], %[vl]\n\t"
            "j 1b\n\t"
            "2:\n\t"
            : [sum] "+f"(sum),
              [chunk] "=&f"(chunk),
              [x] "+r"(px),
              [remaining] "+r"(remaining),
              [vl] "=&r"(vl),
              [bytes] "=&r"(bytes)
            : [zero] "f"(zero)
            : "memory"
        );

        return sum;
    }
#endif

    T sum{};
    for (std::size_t i = 0; i < n; ++i) {
        sum += x[i];
    }
    return sum;
}

template <class T>
void affine_raw(const T* x, T* out, std::size_t n, T scale, T offset) {
#if defined(__riscv_vector)
    if constexpr (std::is_same_v<std::remove_cv_t<T>, float>) {
        auto* x_iter = reinterpret_cast<const float*>(x);
        auto* out_iter = reinterpret_cast<float*>(out);
        std::size_t remaining = n;
        std::size_t vl = 0;
        std::size_t bytes = 0;

        asm volatile(
            "1:\n\t"
            "beqz %[remaining], 2f\n\t"
            "vsetvli %[vl], %[remaining], e32, m1, ta, ma\n\t"
            "vle32.v v0, (%[x])\n\t"
            "vfmul.vf v0, v0, %[scale]\n\t"
            "vfadd.vf v0, v0, %[offset]\n\t"
            "vse32.v v0, (%[out])\n\t"
            "slli %[bytes], %[vl], 2\n\t"
            "add %[x], %[x], %[bytes]\n\t"
            "add %[out], %[out], %[bytes]\n\t"
            "sub %[remaining], %[remaining], %[vl]\n\t"
            "j 1b\n\t"
            "2:\n\t"
            : [x] "+r"(x_iter),
              [out] "+r"(out_iter),
              [remaining] "+r"(remaining),
              [vl] "=&r"(vl),
              [bytes] "=&r"(bytes)
            : [scale] "f"(scale),
              [offset] "f"(offset)
            : "memory"
        );
        return;
    }
#endif

    for (std::size_t i = 0; i < n; ++i) {
        out[i] = (x[i] * scale) + offset;
    }
}

template <class T>
void gemv_raw(const T* matrix, const T* x, T* out, std::size_t rows, std::size_t cols) {
#if defined(__riscv_vector)
    if constexpr (std::is_same_v<std::remove_cv_t<T>, float>) {
        const float zero = 0.0f;
        const std::size_t stride = cols * sizeof(float);

        for (std::size_t row = 0; row < rows;) {
            std::size_t vl = 0;
            const float* row_base = reinterpret_cast<const float*>(matrix + (row * cols));
            float* out_base = reinterpret_cast<float*>(out + row);
            const float* x_iter = nullptr;
            const float* col_iter = nullptr;
            std::size_t k = 0;
            float x_value = 0.0f;

            asm volatile(
                "vsetvli %[vl], %[rows_left], e32, m1, ta, ma\n\t"
                "vfmv.v.f v0, %[zero]\n\t"
                "mv %[x_iter], %[x]\n\t"
                "mv %[col_iter], %[row_base]\n\t"
                "mv %[k], %[cols]\n\t"
                "1:\n\t"
                "beqz %[k], 2f\n\t"
                "flw %[x_value], 0(%[x_iter])\n\t"
                "vlse32.v v1, (%[col_iter]), %[stride]\n\t"
                "vfmacc.vf v0, %[x_value], v1\n\t"
                "addi %[x_iter], %[x_iter], 4\n\t"
                "addi %[col_iter], %[col_iter], 4\n\t"
                "addi %[k], %[k], -1\n\t"
                "j 1b\n\t"
                "2:\n\t"
                "vse32.v v0, (%[out_base])\n\t"
                : [vl] "=&r"(vl),
                  [x_iter] "=&r"(x_iter),
                  [col_iter] "=&r"(col_iter),
                  [k] "=&r"(k),
                  [x_value] "=&f"(x_value)
                : [rows_left] "r"(rows - row),
                  [x] "r"(x),
                  [row_base] "r"(row_base),
                  [cols] "r"(cols),
                  [stride] "r"(stride),
                  [out_base] "r"(out_base),
                  [zero] "f"(zero)
                : "memory"
            );

            row += vl;
        }
        return;
    }
#endif

    for (std::size_t row = 0; row < rows; ++row) {
        T acc{};
        const T* matrix_row = matrix + (row * cols);
        for (std::size_t col = 0; col < cols; ++col) {
            acc += matrix_row[col] * x[col];
        }
        out[row] = acc;
    }
}

template <Vector1D VA, Vector1D VB>
requires std::same_as<
    std::remove_cv_t<typename VA::element_type>,
    std::remove_cv_t<typename VB::element_type>>
auto dot(const VA& a, const VB& b) -> std::remove_cv_t<typename VA::element_type> {
    using T = std::remove_cv_t<typename VA::element_type>;
    constexpr std::size_t N = VA::static_extent(0);

    static_assert(VB::static_extent(0) == N, "dot: vector lengths must match");

    return dot_raw<T>(a.data(), b.data(), N);
}

template <Matrix2D MA, Vector1D VX, Vector1D VY>
requires std::same_as<
             std::remove_cv_t<typename MA::element_type>,
             std::remove_cv_t<typename VX::element_type>> &&
         std::same_as<
             std::remove_cv_t<typename MA::element_type>,
             std::remove_cv_t<typename VY::element_type>> &&
         (!std::is_const_v<typename VY::element_type>)
void gemv(const MA& a, const VX& x, VY& y) {
    using T = std::remove_cv_t<typename MA::element_type>;
    constexpr std::size_t M = MA::static_extent(0);
    constexpr std::size_t N = MA::static_extent(1);

    static_assert(VX::static_extent(0) == N, "gemv: x length must equal A columns");
    static_assert(VY::static_extent(0) == M, "gemv: y length must equal A rows");

    gemv_raw<T>(a.data(), x.data(), y.data(), M, N);
}

template <class T>
T mdot(const T* a, const T* b, std::size_t n) {
    return dot_raw<T>(a, b, n);
}

template <class T>
void mgemv(const T* matrix, const T* x, T* out, std::size_t rows, std::size_t cols) {
    gemv_raw<T>(matrix, x, out, rows, cols);
}

}  // namespace blas

#endif
