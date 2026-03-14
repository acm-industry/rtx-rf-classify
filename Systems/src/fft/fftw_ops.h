#ifndef __FFTW_OPS_H__
#define __FFTW_OPS_H__

#include <type_traits>
#include <complex>
#include <concepts>
#include <fftw3.h>
#include "../tensor.h"
#include "fftw_wrapper.h"

namespace fft {

// Type trait mapping float/double to their respective FFTW configurations
template <typename T>
struct fftw_traits;

template <>
struct fftw_traits<float> {
    using plan_t = fftwf_plan;
    using complex_t = fftwf_complex;
    static constexpr auto plan_dft = fftwf_plan_dft;
    static constexpr auto execute = fftwf_execute;
    static constexpr auto execute_dft = fftwf_execute_dft;
    static constexpr auto destroy_plan = fftwf_destroy_plan;
};

template <>
struct fftw_traits<double> {
    using plan_t = fftw_plan;
    using complex_t = fftw_complex;
    static constexpr auto plan_dft = fftw_plan_dft;
    static constexpr auto execute = fftw_execute;
    static constexpr auto execute_dft = fftw_execute_dft;
    static constexpr auto destroy_plan = fftw_destroy_plan;
};

// Also support std::complex mapping
template <typename T>
struct fftw_traits<std::complex<T>> : fftw_traits<T> {};

/**
 * Type-safe FFTW wrapper using TensorBase with compile-time extent checking
 * Provides memory management of plans and safe `operator()` overloading for evaluation
 */
template <typename T, FixedExtent E, int Sign = FFTW_FORWARD, unsigned Flags = FFTW_ESTIMATE>
class FFTW {
public:
    using traits = fftw_traits<T>;
    using plan_t = typename traits::plan_t;
    using in_t = std::complex<typename std::conditional_t<std::is_same_v<T, float>, float, double>>;
    // T could be float or std::complex<float>, we normalize it back to complex since FFTW expects complex.
    using val_t = std::conditional_t<std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>, T, std::complex<T>>;

private:
    plan_t plan_{nullptr};
    bool valid_{false};

    // Helper to generate the plan
    template <Allocator<val_t> AllocIn, Allocator<val_t> AllocOut>
    void create_plan(DynTensor<val_t, E, AllocIn>& in, DynTensor<val_t, E, AllocOut>& out) {
        if (valid_) return;

        // Since it's a fixed extent, we extract dimensions at compile time
        // Note: For multi-dimensional, we can pass multiple dimensions, but here we assume rank 1 for simplicity of FFTW plan. 
        // A full rank-N FFT would require fftw_plan_dft_X. For now, assuming rank 1 as a start.
        static_assert(E::rank() == 1, "Currently only rank 1 tensors are supported for FFTW");
        
        int n = E::static_extent(0);
        plan_ = traits::plan_dft(
            1, &n, 
            reinterpret_cast<typename traits::complex_t*>(in.data()), 
            reinterpret_cast<typename traits::complex_t*>(out.data()), 
            Sign, Flags
        );
        valid_ = plan_ != nullptr;
    }

public:
    FFTW() = default;
    
    // Disable copy
    FFTW(const FFTW&) = delete;
    FFTW& operator=(const FFTW&) = delete;

    // Enable move
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

    /**
     * Executes the FFT plan using in and out tensors.
     * Tensors must have the exact same rank and extents as established by FixedExtent E.
     */
    template <Allocator<val_t> AllocIn, Allocator<val_t> AllocOut>
    void operator()(DynTensor<val_t, E, AllocIn>& in, DynTensor<val_t, E, AllocOut>& out) {
        if (!valid_) {
            create_plan(in, out);
        }
        
        // Execute the plan
        // If pointers match the one generated at planning time, `execute` is perfectly safe
        // If not, execute_dft must be used for new array pointers
        traits::execute_dft(plan_, 
            reinterpret_cast<typename traits::complex_t*>(in.data()), 
            reinterpret_cast<typename traits::complex_t*>(out.data()));
    }

    /**
     * Integration with the expression system.
     * Materializes the expression into `temp_in`, then performs the FFT, saving into `out`.
     */
    template <Expression Expr, Allocator<val_t> AllocIn, Allocator<val_t> AllocOut>
    void eval(const Expr& expr, DynTensor<val_t, E, AllocIn>& temp_in, DynTensor<val_t, E, AllocOut>& out) {
        static_assert(Expr::iter_size() == compute_static_size<E>(),
            "FFTW eval: expression size must match tensor size");

        for (size_t i = 0; i < Expr::iter_size(); ++i) {
            temp_in.flat(i) = expr.access(i);
        }

        this->operator()(temp_in, out);
    }
};

} // namespace fft

#endif
