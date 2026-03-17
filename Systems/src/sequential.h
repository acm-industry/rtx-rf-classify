#ifndef __SEQUENTIAL_H__
#define __SEQUENTIAL_H__

#include <algorithm>
#include <cstddef>
#include <memory>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>

#include "tensor.h"

namespace sequential_detail {

    // Extract `InTensor` and `OutTensor` from a layer signature:
    //   void operator()(const InTensor&, OutTensor&) [const]
    template <class>
    struct layer_traits;

    template <class C, class In, class Out>
    struct layer_traits<void (C::*)(const In&, Out&) const> {
        using in_tensor = std::remove_cvref_t<In>;
        using out_tensor = std::remove_cvref_t<Out>;
    };

    template <class C, class In, class Out>
    struct layer_traits<void (C::*)(const In&, Out&)> {
        using in_tensor = std::remove_cvref_t<In>;
        using out_tensor = std::remove_cvref_t<Out>;
    };

    template <class Layer>
    using layer_traits_t = layer_traits<decltype(&Layer::operator())>;

    template <class Layer>
    using layer_in_t = typename layer_traits_t<Layer>::in_tensor;

    template <class Layer>
    using layer_out_t = typename layer_traits_t<Layer>::out_tensor;

    template <class Alloc>
    struct scratch_buffer {
        using byte_alloc = typename std::allocator_traits<
            Alloc>::template rebind_alloc<std::byte>;

        byte_alloc alloc;
        std::byte* raw{};
        std::byte* aligned{};
        std::size_t raw_bytes{};
        std::size_t used_bytes{};
        std::size_t alignment{};

        scratch_buffer(const Alloc& base, std::size_t bytes, std::size_t align)
            : alloc(base),
              raw_bytes(bytes + (align ? (align - 1) : 0)),
              used_bytes(bytes),
              alignment(align) {
            if (used_bytes == 0) return;
            raw = std::allocator_traits<byte_alloc>::allocate(alloc, raw_bytes);

            void* p = raw;
            std::size_t space = raw_bytes;
            void* ap = std::align(alignment, used_bytes, p, space);
            aligned = static_cast<std::byte*>(ap);
        }

        scratch_buffer(const scratch_buffer&) = delete;
        scratch_buffer& operator=(const scratch_buffer&) = delete;

        scratch_buffer(scratch_buffer&& other) noexcept
            : alloc(std::move(other.alloc)),
              raw(std::exchange(other.raw, nullptr)),
              aligned(std::exchange(other.aligned, nullptr)),
              raw_bytes(std::exchange(other.raw_bytes, 0)),
              used_bytes(std::exchange(other.used_bytes, 0)),
              alignment(std::exchange(other.alignment, 0)) {}

        scratch_buffer& operator=(scratch_buffer&& other) noexcept {
            if (this == &other) return *this;
            if (raw != nullptr) {
                std::allocator_traits<byte_alloc>::deallocate(
                    alloc, raw, raw_bytes
                );
            }
            alloc = std::move(other.alloc);
            raw = std::exchange(other.raw, nullptr);
            aligned = std::exchange(other.aligned, nullptr);
            raw_bytes = std::exchange(other.raw_bytes, 0);
            used_bytes = std::exchange(other.used_bytes, 0);
            alignment = std::exchange(other.alignment, 0);
            return *this;
        }

        ~scratch_buffer() {
            if (raw == nullptr) return;
            std::allocator_traits<byte_alloc>::deallocate(
                alloc, raw, raw_bytes
            );
        }
    };

}  // namespace sequential_detail

template <class A, class... Layers>
    requires(sizeof...(Layers) > 0) && Allocator<A, std::byte>
class Sequential {
    using layers_tuple_t = std::tuple<Layers...>;

    template <std::size_t I>
    using layer_t = std::tuple_element_t<I, layers_tuple_t>;

    static constexpr std::size_t n_layers = sizeof...(Layers);

    template <class T>
    struct is_tensor_base : std::false_type {};

    template <class T, FixedExtent E>
    struct is_tensor_base<TensorBase<T, E>> : std::true_type {};

    template <class T>
    static constexpr bool is_tensor_base_v =
        is_tensor_base<std::remove_cvref_t<T>>::value;

    template <class TB>
    static consteval std::size_t tensor_bytes() {
        using tensor_t = std::remove_cvref_t<TB>;
        return sizeof(typename tensor_t::value_type) * tensor_t::static_size;
    }

    template <class TB>
    static consteval std::size_t tensor_alignment() {
        using tensor_t = std::remove_cvref_t<TB>;
        return alignof(typename tensor_t::value_type);
    }

    template <class TB>
    static constexpr TB make_tensor_view(std::byte* buffer) noexcept {
        using tensor_t = std::remove_cvref_t<TB>;
        using value_t = typename tensor_t::value_type;
        constexpr std::size_t n = tensor_t::static_size;

        auto* typed = reinterpret_cast<value_t*>(buffer);
        return tensor_t(std::span<value_t, n>(typed, n));
    }

    template <class L>
    static consteval void validate_layer() {
        using in_t = sequential_detail::layer_in_t<L>;
        using out_t = sequential_detail::layer_out_t<L>;

        static_assert(
            is_tensor_base_v<in_t>,
            "Sequential layer input must be a TensorBase<T, Extents> type."
        );
        static_assert(
            is_tensor_base_v<out_t>,
            "Sequential layer output must be a TensorBase<T, Extents> type."
        );
        static_assert(
            !std::is_const_v<typename out_t::value_type>,
            "Sequential layer output tensor value_type must be non-const."
        );
    }

    template <class ALayer, class BLayer>
    static consteval bool adjacent_match() {
        using a_out = sequential_detail::layer_out_t<ALayer>;
        using b_in = sequential_detail::layer_in_t<BLayer>;

        using a_value = std::remove_const_t<typename a_out::value_type>;
        using b_value = std::remove_const_t<typename b_in::value_type>;

        return std::is_same_v<a_value, b_value> &&
               (a_out::static_size == b_in::static_size);
    }

    template <std::size_t I>
    static consteval void validate_chain() {
        if constexpr (I + 1 < n_layers) {
            static_assert(
                adjacent_match<layer_t<I>, layer_t<I + 1>>(),
                "Sequential pipeline mismatch: layer i output must match layer "
                "i+1 input "
                "(same value_type (ignoring const) and same static_size)."
            );
            validate_chain<I + 1>();
        }
    }

    static consteval bool validate_pipeline() {
        (validate_layer<Layers>(), ...);
        validate_chain<0>();
        return true;
    }

    static_assert(validate_pipeline());

    static consteval std::size_t compute_scratch_bytes() {
        if constexpr (n_layers <= 1) return 0;
        std::size_t max_bytes = 0;
        [&]<std::size_t... Is>(std::index_sequence<Is...>) consteval {
            ((max_bytes = std::max(
                  max_bytes,
                  tensor_bytes<sequential_detail::layer_out_t<layer_t<Is>>>()
              )),
             ...);
        }(
            std::make_index_sequence<n_layers - 1>{}
        );  // intermediate outputs: [0..n-2]
        return max_bytes;
    }

    static consteval std::size_t compute_scratch_alignment() {
        if constexpr (n_layers <= 1) return 1;
        std::size_t max_align = 1;
        [&]<std::size_t... Is>(std::index_sequence<Is...>) consteval {
            ((max_align = std::max(
                  max_align,
                  tensor_alignment<sequential_detail::layer_out_t<layer_t<Is>>>()
              )),
             ...);
        }(
            std::make_index_sequence<n_layers - 1>{}
        );  // intermediate outputs: [0..n-2]
        return max_align;
    }

   public:
    using allocator_type = A;
    using input_tensor_type = sequential_detail::layer_in_t<layer_t<0>>;
    using output_tensor_type =
        sequential_detail::layer_out_t<layer_t<n_layers - 1>>;

    static constexpr std::size_t scratch_bytes = compute_scratch_bytes();
    static constexpr std::size_t scratch_alignment =
        compute_scratch_alignment();

   private:
    [[no_unique_address]] mutable A allocator_;
    [[no_unique_address]] layers_tuple_t layers_;

    template <std::size_t I>
    constexpr decltype(auto) layer() noexcept {
        return std::get<I>(layers_);
    }

    template <std::size_t I>
    constexpr decltype(auto) layer() const noexcept {
        return std::get<I>(layers_);
    }

    template <std::size_t I>
    using in_t = sequential_detail::layer_in_t<layer_t<I>>;

    template <std::size_t I>
    using out_t = sequential_detail::layer_out_t<layer_t<I>>;

    template <std::size_t I>
    constexpr void run_stage(std::byte* in_buf, std::byte* out_buf) const {
        auto in_view = make_tensor_view<in_t<I>>(in_buf);
        auto out_view = make_tensor_view<out_t<I>>(out_buf);
        layer<I>()(in_view, out_view);
    }

   public:
    constexpr Sequential() = default;

    template <class... LayerInits>
        requires(sizeof...(LayerInits) == sizeof...(Layers) &&
                 (std::constructible_from<Layers, LayerInits &&> && ...))
    constexpr explicit Sequential(const A& allocator, LayerInits&&... layers)
        : allocator_(allocator),
          layers_(Layers(std::forward<LayerInits>(layers))...) {}

    constexpr const A& allocator() const noexcept { return allocator_; }

    constexpr void operator()(
        const input_tensor_type& in, output_tensor_type& out
    ) const {
        if constexpr (n_layers == 1) {
            layer<0>()(in, out);
            return;
        }

        sequential_detail::scratch_buffer<A> scratch1(
            allocator_, scratch_bytes, scratch_alignment
        );
        sequential_detail::scratch_buffer<A> scratch2(
            allocator_, scratch_bytes, scratch_alignment
        );

        std::byte* ping = scratch1.aligned;
        std::byte* pong = scratch2.aligned;

        // Stage 0: in -> ping
        {
            auto out_view = make_tensor_view<out_t<0>>(ping);
            layer<0>()(in, out_view);
        }

        // Middle stages: ping <-> pong
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            // Is maps to stage index (1..n_layers-2)
            (([&] {
                 constexpr std::size_t stage = Is + 1;
                 run_stage<stage>(ping, pong);
                 std::swap(ping, pong);
             }()),
             ...);
        }(std::make_index_sequence<n_layers - 2>{});

        // Final stage: ping -> out
        {
            auto in_view = make_tensor_view<in_t<n_layers - 1>>(ping);
            layer<n_layers - 1>()(in_view, out);
        }
    }
};

#endif
