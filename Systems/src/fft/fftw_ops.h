#ifndef __FFTW_OPS_H__
#define __FFTW_OPS_H__

#include <type_traits>
#include <complex>
#include <concepts>
#include <array>
#include <fftw3.h>
#include "../tensor.h"
#include "fftw_wrapper.h"

namespace fft {

// Type trait mapping float/double to their respective FFTW configurations
template <typename T>
struct fftw_traits;

template <>
struct fftw_traits<float> {
    using real_t = float;
    using plan_t = fftwf_plan;
    using complex_t = fftwf_complex;
    static constexpr auto plan_dft = fftwf_plan_dft;
    static constexpr auto plan_dft_r2c = fftwf_plan_dft_r2c_1d;
    static constexpr auto plan_dft_c2r = fftwf_plan_dft_c2r_1d;
    static constexpr auto execute = fftwf_execute;
    static constexpr auto execute_dft = fftwf_execute_dft;
    static constexpr auto execute_dft_r2c = fftwf_execute_dft_r2c;
    static constexpr auto execute_dft_c2r = fftwf_execute_dft_c2r;
    static constexpr auto destroy_plan = fftwf_destroy_plan;
};

template <>
struct fftw_traits<double> {
    using real_t = double;
    using plan_t = fftw_plan;
    using complex_t = fftw_complex;
    static constexpr auto plan_dft = fftw_plan_dft;
    static constexpr auto plan_dft_r2c = fftw_plan_dft_r2c_1d;
    static constexpr auto plan_dft_c2r = fftw_plan_dft_c2r_1d;
    static constexpr auto execute = fftw_execute;
    static constexpr auto execute_dft = fftw_execute_dft;
    static constexpr auto execute_dft_r2c = fftw_execute_dft_r2c;
    static constexpr auto execute_dft_c2r = fftw_execute_dft_c2r;
    static constexpr auto destroy_plan = fftw_destroy_plan;
};

// Also support std::complex mapping
template <typename T>
struct fftw_traits<std::complex<T>> : fftw_traits<T> {};

// Helper to compute R2C Output Extent
template <FixedExtent E>
using R2CExtent = std::extents<size_t, (E::static_extent(0) / 2) + 1>;

/**
 * Type-safe FFTW wrapper using TensorBase with compile-time extent checking
 * Provides memory management of plans and safe `operator()` overloading for evaluation
 */
template <typename T, FixedExtent E, int Sign = FFTW_FORWARD, unsigned Flags = FFTW_ESTIMATE>
class FFTW {
public:
    using traits = fftw_traits<T>;
    using plan_t = typename traits::plan_t;
    using val_t = std::complex<typename traits::real_t>;

private:
    plan_t plan_{nullptr};
    bool valid_{false};

    void create_plan(TensorBase<val_t, E>& in, TensorBase<val_t, E>& out) {
        if (valid_) return;

        std::array<int, E::rank()> n{};
        for (std::size_t i = 0; i < E::rank(); ++i) {
            n[i] = static_cast<int>(E::static_extent(i));
        }

        plan_ = traits::plan_dft(
            static_cast<int>(E::rank()), n.data(),
            reinterpret_cast<typename traits::complex_t*>(in.data()),
            reinterpret_cast<typename traits::complex_t*>(out.data()),
            Sign, Flags
        );
        valid_ = plan_ != nullptr;
    }

public:
    FFTW() = default;
    FFTW(const FFTW&) = delete;
    FFTW& operator=(const FFTW&) = delete;

    FFTW(FFTW&& other) noexcept : plan_(other.plan_), valid_(other.valid_) {
        other.plan_ = nullptr;
        other.valid_ = false;
    }
    FFTW& operator=(FFTW&& other) noexcept {
        if (this != &other) {
            if (valid_) traits::destroy_plan(plan_);
            plan_ = other.plan_;
            valid_ = other.valid_;
            other.plan_ = nullptr;
            other.valid_ = false;
        }
        return *this;
    }

    ~FFTW() {
        if (valid_) traits::destroy_plan(plan_);
    }

    void operator()(TensorBase<val_t, E>& in, TensorBase<val_t, E>& out) {
        if (!valid_) {
            create_plan(in, out);
        }
        traits::execute_dft(plan_, 
            reinterpret_cast<typename traits::complex_t*>(in.data()), 
            reinterpret_cast<typename traits::complex_t*>(out.data()));
    }

    template <Expression Expr>
    void eval(const Expr& expr, TensorBase<val_t, E>& temp_in, TensorBase<val_t, E>& out) {
        static_assert(Expr::iter_size() == compute_static_size<E>(),
            "FFTW eval: expression size must match tensor size");

        for (size_t i = 0; i < Expr::iter_size(); ++i) {
            temp_in.flat(i) = expr.access(i);
        }
        this->operator()(temp_in, out);
    }
};

/**
 * Real-to-Complex Transform
 * Input is a real Tensor, Output is a complex Tensor of size N/2 + 1.
 */
template <typename T, FixedExtent E, unsigned Flags = FFTW_ESTIMATE>
class FFTW_R2C {
public:
    using traits = fftw_traits<T>;
    using plan_t = typename traits::plan_t;
    using real_t = typename traits::real_t;
    using complex_t = std::complex<real_t>;
    using OutExtent = R2CExtent<E>;

private:
    plan_t plan_{nullptr};
    bool valid_{false};

    void create_plan(TensorBase<real_t, E>& in, TensorBase<complex_t, OutExtent>& out) {
        if (valid_) return;
        static_assert(E::rank() == 1, "Currently only rank 1 tensors are supported for FFTW_R2C");
        
        int n = E::static_extent(0);
        plan_ = traits::plan_dft_r2c(
            n, 
            reinterpret_cast<real_t*>(in.data()), 
            reinterpret_cast<typename traits::complex_t*>(out.data()), 
            Flags
        );
        valid_ = plan_ != nullptr;
    }

public:
    FFTW_R2C() = default;
    FFTW_R2C(const FFTW_R2C&) = delete;
    FFTW_R2C& operator=(const FFTW_R2C&) = delete;

    FFTW_R2C(FFTW_R2C&& other) noexcept : plan_(other.plan_), valid_(other.valid_) {
        other.plan_ = nullptr;
        other.valid_ = false;
    }
    FFTW_R2C& operator=(FFTW_R2C&& other) noexcept {
        if (this != &other) {
            if (valid_) traits::destroy_plan(plan_);
            plan_ = other.plan_;
            valid_ = other.valid_;
            other.plan_ = nullptr;
            other.valid_ = false;
        }
        return *this;
    }

    ~FFTW_R2C() {
        if (valid_) traits::destroy_plan(plan_);
    }

    void operator()(TensorBase<real_t, E>& in, TensorBase<complex_t, OutExtent>& out) {
        if (!valid_) {
            create_plan(in, out);
        }
        traits::execute_dft_r2c(plan_, 
            reinterpret_cast<real_t*>(in.data()), 
            reinterpret_cast<typename traits::complex_t*>(out.data()));
    }

    template <Expression Expr>
    void eval(const Expr& expr, TensorBase<real_t, E>& temp_in, TensorBase<complex_t, OutExtent>& out) {
        static_assert(Expr::iter_size() == compute_static_size<E>(),
            "FFTW_R2C eval: expression size must match tensor size");

        for (size_t i = 0; i < Expr::iter_size(); ++i) {
            temp_in.flat(i) = expr.access(i);
        }
        this->operator()(temp_in, out);
    }
};

/**
 * Complex-to-Real Transform
 * Input is a complex Tensor of size N/2 + 1, Output is a real Tensor of size N.
 */
template <typename T, FixedExtent OutE, unsigned Flags = FFTW_ESTIMATE>
class FFTW_C2R {
public:
    using traits = fftw_traits<T>;
    using plan_t = typename traits::plan_t;
    using real_t = typename traits::real_t;
    using complex_t = std::complex<real_t>;
    using InExtent = R2CExtent<OutE>;

private:
    plan_t plan_{nullptr};
    bool valid_{false};

    void create_plan(TensorBase<complex_t, InExtent>& in, TensorBase<real_t, OutE>& out) {
        if (valid_) return;
        static_assert(OutE::rank() == 1, "Currently only rank 1 tensors are supported for FFTW_C2R");
        
        int n = OutE::static_extent(0);
        plan_ = traits::plan_dft_c2r(
            n, 
            reinterpret_cast<typename traits::complex_t*>(in.data()), 
            reinterpret_cast<real_t*>(out.data()), 
            Flags
        );
        valid_ = plan_ != nullptr;
    }

public:
    FFTW_C2R() = default;
    FFTW_C2R(const FFTW_C2R&) = delete;
    FFTW_C2R& operator=(const FFTW_C2R&) = delete;

    FFTW_C2R(FFTW_C2R&& other) noexcept : plan_(other.plan_), valid_(other.valid_) {
        other.plan_ = nullptr;
        other.valid_ = false;
    }
    FFTW_C2R& operator=(FFTW_C2R&& other) noexcept {
        if (this != &other) {
            if (valid_) traits::destroy_plan(plan_);
            plan_ = other.plan_;
            valid_ = other.valid_;
            other.plan_ = nullptr;
            other.valid_ = false;
        }
        return *this;
    }

    ~FFTW_C2R() {
        if (valid_) traits::destroy_plan(plan_);
    }

    void operator()(TensorBase<complex_t, InExtent>& in, TensorBase<real_t, OutE>& out) {
        if (!valid_) {
            create_plan(in, out);
        }
        traits::execute_dft_c2r(plan_, 
            reinterpret_cast<typename traits::complex_t*>(in.data()), 
            reinterpret_cast<real_t*>(out.data()));
    }

    template <Expression Expr>
    void eval(const Expr& expr, TensorBase<complex_t, InExtent>& temp_in, TensorBase<real_t, OutE>& out) {
        static_assert(Expr::iter_size() == compute_static_size<InExtent>(),
            "FFTW_C2R eval: expression size must match tensor size");

        for (size_t i = 0; i < Expr::iter_size(); ++i) {
            temp_in.flat(i) = expr.access(i);
        }
        this->operator()(temp_in, out);
    }
};

} // namespace fft

#endif
