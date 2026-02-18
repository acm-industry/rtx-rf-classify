#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <random>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "../memorybuffer.h"
#include "../tensor.h"

#ifdef USE_FFTW
#include "../fft/fftw_wrapper.h"
#include <fftw3.h>
#endif

namespace cnn {

inline void init_random(float* ptr, std::size_t n, float scale = 0.1f) {
    static std::mt19937 rng(12345);
    static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (std::size_t i = 0; i < n; ++i) {
        ptr[i] = dist(rng) * scale;
    }
}

constexpr std::size_t next_pow2(std::size_t n) {
    std::size_t p = 1;
    while (p < n) {
        p <<= 1;
    }
    return p;
}

bool fftw_backend_available();

// Dense linear layer: y = W x + b
// W shape = [OutN, InN], x shape = [InN], y shape = [OutN]
template <std::size_t InN, std::size_t OutN, class Alloc>
struct Linear {
    using InTensor = TensorBase<float, std::extents<std::size_t, InN>, Alloc>;
    using OutTensor = TensorBase<float, std::extents<std::size_t, OutN>, Alloc>;
    using WeightTensor = TensorBase<float, std::extents<std::size_t, OutN, InN>, Alloc>;

    WeightTensor weights;
    OutTensor bias;
    Alloc alloc_;

    explicit Linear(Alloc alloc) : weights(alloc), bias(alloc), alloc_(alloc) {
        const float scale = std::sqrt(2.0f / static_cast<float>(InN + OutN));
        init_random(weights.data(), WeightTensor::size, scale);
        std::fill_n(bias.data(), OutN, 0.0f);
    }

    OutTensor operator()(const InTensor& in) const {
        OutTensor out(alloc_);
        for (std::size_t o = 0; o < OutN; ++o) {
            float acc = bias.flat(o);
            for (std::size_t i = 0; i < InN; ++i) {
                acc += weights.flat(o * InN + i) * in.flat(i);
            }
            out.flat(o) = acc;
        }
        return out;
    }
};

struct ReLU {
    template <typename Tensor>
    Tensor operator()(Tensor t) const {
        for (std::size_t i = 0; i < Tensor::size; ++i) {
            t.flat(i) = std::max(t.flat(i), 0.0f);
        }
        return t;
    }
};

// Flatten [C, L] -> [C * L]
template <std::size_t C, std::size_t L, class Alloc>
struct Flatten1D {
    using InTensor = TensorBase<float, std::extents<std::size_t, C, L>, Alloc>;
    using OutTensor = TensorBase<float, std::extents<std::size_t, C * L>, Alloc>;
    Alloc alloc_;

    explicit Flatten1D(Alloc alloc) : alloc_(alloc) {}

    OutTensor operator()(const InTensor& in) const {
        OutTensor out(alloc_);
        for (std::size_t i = 0; i < C * L; ++i) {
            out.flat(i) = in.flat(i);
        }
        return out;
    }
};

// Conv1D for valid cross-correlation (PyTorch-style Conv1d default behavior).
// in shape:  [C_in, L_in]
// out shape: [C_out, L_in - K + 1]
template <std::size_t C_in, std::size_t C_out, std::size_t L_in, std::size_t K, class Alloc>
struct Conv1D {
    static_assert(K > 0, "kernel size must be > 0");
    static_assert(L_in >= K, "input length must be >= kernel size");

    static constexpr std::size_t L_out = L_in - K + 1;

    using InTensor = TensorBase<float, std::extents<std::size_t, C_in, L_in>, Alloc>;
    using OutTensor = TensorBase<float, std::extents<std::size_t, C_out, L_out>, Alloc>;
    using KernelTensor = TensorBase<float, std::extents<std::size_t, C_out, C_in, K>, Alloc>;
    using BiasTensor = TensorBase<float, std::extents<std::size_t, C_out>, Alloc>;

    KernelTensor kernels;
    BiasTensor bias;
    Alloc alloc_;
#ifdef USE_FFTW
    struct FFTCache {
        static constexpr std::size_t full_len = L_in + K - 1;
        static constexpr std::size_t fft_len = next_pow2(full_len);

        fftw_complex* time_in = nullptr;
        fftw_complex* freq_in = nullptr;
        fftw_complex* freq_out = nullptr;
        fftw_complex* time_out = nullptr;
        fftw_complex* time_kernel = nullptr;
        fftw_complex* freq_kernel = nullptr;
        fft::FFTPlan fwd_in;
        fft::FFTPlan fwd_kernel;
        fft::FFTPlan inv_out;
        std::vector<double> kernel_freq_re;
        std::vector<double> kernel_freq_im;
        bool ok = false;

        FFTCache()
            : kernel_freq_re(C_out * C_in * fft_len, 0.0),
              kernel_freq_im(C_out * C_in * fft_len, 0.0) {
            time_in = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * fft_len));
            freq_in = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * fft_len));
            freq_out = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * fft_len));
            time_out = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * fft_len));
            time_kernel = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * fft_len));
            freq_kernel = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * fft_len));

            if (!time_in || !freq_in || !freq_out || !time_out || !time_kernel || !freq_kernel) {
                return;
            }

            fwd_in = fft::PlanFactory::create_1d(static_cast<int>(fft_len), time_in, freq_in,
                                                 FFTW_FORWARD, FFTW_ESTIMATE);
            fwd_kernel = fft::PlanFactory::create_1d(static_cast<int>(fft_len), time_kernel,
                                                     freq_kernel, FFTW_FORWARD, FFTW_ESTIMATE);
            inv_out = fft::PlanFactory::create_1d(static_cast<int>(fft_len), freq_out, time_out,
                                                  FFTW_BACKWARD, FFTW_ESTIMATE);
            ok = fwd_in.valid() && fwd_kernel.valid() && inv_out.valid();
        }

        ~FFTCache() {
            if (time_in) fftw_free(time_in);
            if (freq_in) fftw_free(freq_in);
            if (freq_out) fftw_free(freq_out);
            if (time_out) fftw_free(time_out);
            if (time_kernel) fftw_free(time_kernel);
            if (freq_kernel) fftw_free(freq_kernel);
        }
    };

    mutable std::shared_ptr<FFTCache> fft_cache_;
#endif

    explicit Conv1D(Alloc alloc) : kernels(alloc), bias(alloc), alloc_(alloc) {
        const float scale = std::sqrt(2.0f / static_cast<float>(C_in * K + C_out * K));
        init_random(kernels.data(), KernelTensor::size, scale);
        std::fill_n(bias.data(), C_out, 0.0f);
    }

    // Call this after manually changing kernels/bias so FFT kernel cache is rebuilt.
    void invalidate_fft_cache() const {
#ifdef USE_FFTW
        fft_cache_.reset();
#endif
    }

    OutTensor operator()(const InTensor& in) const {
        OutTensor out(alloc_);
        for (std::size_t o = 0; o < C_out; ++o) {
            for (std::size_t x = 0; x < L_out; ++x) {
                out.flat(o * L_out + x) = bias.flat(o);
            }
        }

        if (should_use_fft() && fft_accumulate_all(in, out)) {
            return out;
        }

        for (std::size_t o = 0; o < C_out; ++o) {
            for (std::size_t i_ch = 0; i_ch < C_in; ++i_ch) {
                direct_accumulate_channel(in, out, o, i_ch);
            }
        }

        return out;
    }

  private:
    static constexpr bool should_use_fft() {
#ifdef USE_FFTW
        // A conservative crossover heuristic; direct conv is better on tiny kernels.
        return (L_in * K >= 256);
#else
        return false;
#endif
    }

    void direct_accumulate_channel(const InTensor& in, OutTensor& out, std::size_t out_ch,
                                   std::size_t in_ch) const {
        for (std::size_t x = 0; x < L_out; ++x) {
            float acc = 0.0f;
            for (std::size_t k = 0; k < K; ++k) {
                const std::size_t in_idx = in_ch * L_in + (x + k);
                const std::size_t ker_idx = (out_ch * C_in + in_ch) * K + k;
                acc += in.flat(in_idx) * kernels.flat(ker_idx);
            }
            out.flat(out_ch * L_out + x) += acc;
        }
    }

    bool fft_accumulate_all(const InTensor& in, OutTensor& out) const {
#ifndef USE_FFTW
        (void)in;
        (void)out;
        return false;
#else
        auto cache = get_or_build_fft_cache();
        if (!cache || !cache->ok) {
            return false;
        }

        constexpr std::size_t fft_len = FFTCache::fft_len;
        std::vector<double> in_freq_re(C_in * fft_len, 0.0);
        std::vector<double> in_freq_im(C_in * fft_len, 0.0);

        for (std::size_t i_ch = 0; i_ch < C_in; ++i_ch) {
            for (std::size_t n = 0; n < fft_len; ++n) {
                cache->time_in[n][0] = 0.0;
                cache->time_in[n][1] = 0.0;
            }
            for (std::size_t n = 0; n < L_in; ++n) {
                cache->time_in[n][0] = static_cast<double>(in.flat(i_ch * L_in + n));
            }

            cache->fwd_in.execute();

            const std::size_t base = i_ch * fft_len;
            for (std::size_t n = 0; n < fft_len; ++n) {
                in_freq_re[base + n] = cache->freq_in[n][0];
                in_freq_im[base + n] = cache->freq_in[n][1];
            }
        }

        const double norm = static_cast<double>(fft_len);
        for (std::size_t o = 0; o < C_out; ++o) {
            for (std::size_t i_ch = 0; i_ch < C_in; ++i_ch) {
                const std::size_t in_base = i_ch * fft_len;
                const std::size_t k_base = (o * C_in + i_ch) * fft_len;

                for (std::size_t n = 0; n < fft_len; ++n) {
                    const double ar = in_freq_re[in_base + n];
                    const double ai = in_freq_im[in_base + n];
                    const double br = cache->kernel_freq_re[k_base + n];
                    const double bi = cache->kernel_freq_im[k_base + n];
                    cache->freq_out[n][0] = ar * br - ai * bi;
                    cache->freq_out[n][1] = ar * bi + ai * br;
                }

                cache->inv_out.execute();

                for (std::size_t x = 0; x < L_out; ++x) {
                    const std::size_t idx = x + (K - 1);
                    out.flat(o * L_out + x) += static_cast<float>(cache->time_out[idx][0] / norm);
                }
            }
        }

        return true;
#endif
    }

#ifdef USE_FFTW
    void precompute_fft_kernels(FFTCache& cache) const {
        constexpr std::size_t fft_len = FFTCache::fft_len;

        for (std::size_t o = 0; o < C_out; ++o) {
            for (std::size_t i_ch = 0; i_ch < C_in; ++i_ch) {
                for (std::size_t n = 0; n < fft_len; ++n) {
                    cache.time_kernel[n][0] = 0.0;
                    cache.time_kernel[n][1] = 0.0;
                }

                for (std::size_t k = 0; k < K; ++k) {
                    const float val = kernels.flat((o * C_in + i_ch) * K + (K - 1 - k));
                    cache.time_kernel[k][0] = static_cast<double>(val);
                }

                cache.fwd_kernel.execute();
                const std::size_t base = (o * C_in + i_ch) * fft_len;
                for (std::size_t n = 0; n < fft_len; ++n) {
                    cache.kernel_freq_re[base + n] = cache.freq_kernel[n][0];
                    cache.kernel_freq_im[base + n] = cache.freq_kernel[n][1];
                }
            }
        }
    }

    std::shared_ptr<FFTCache> get_or_build_fft_cache() const {
        if (!fft_cache_) {
            fft_cache_ = std::make_shared<FFTCache>();
            if (fft_cache_->ok) {
                precompute_fft_kernels(*fft_cache_);
            }
        }
        return fft_cache_;
    }
#endif
};

// Compile-time composition of layers.
template <typename... Layers>
struct Sequential {
    std::tuple<Layers...> layers;

    explicit Sequential(Layers... ls) : layers(std::move(ls)...) {}

    template <typename Input>
    auto operator()(Input&& in) {
        return apply_impl<0>(std::forward<Input>(in));
    }

  private:
    template <std::size_t I, typename Input>
    auto apply_impl(Input&& in) -> decltype(auto) {
        if constexpr (I == sizeof...(Layers)) {
            return std::forward<Input>(in);
        } else {
            auto& layer = std::get<I>(layers);
            auto out = layer(std::forward<Input>(in));
            return apply_impl<I + 1>(std::move(out));
        }
    }
};

}
