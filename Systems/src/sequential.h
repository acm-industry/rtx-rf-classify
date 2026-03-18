#ifndef __SEQUENTIAL_H__
#define __SEQUENTIAL_H__

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <memory>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>

#include "tensor.h"

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
using layer_in_t = typename layer_traits<decltype(&Layer::operator())>::in_tensor;

template <class Layer>
using layer_out_t = typename layer_traits<decltype(&Layer::operator())>::out_tensor;

template <class T>
concept StaticTensorView = requires {
    typename T::value_type;
    typename T::extents_type;
    { T::static_size } -> std::convertible_to<std::size_t>;
    { T::byte_size } -> std::convertible_to<std::size_t>;
    { T::alignment } -> std::convertible_to<std::size_t>;
    requires (T::rank > 0);
    requires std::constructible_from<
        T, std::span<typename T::value_type, T::static_size>>;
};

template <class A, class... Layers>
    requires(sizeof...(Layers) > 0) && Allocator<A, std::byte>
class Sequential {
    using layers_types_t = std::tuple<Layers...>;

    template <std::size_t I>
    using layer_t = std::tuple_element_t<I, layers_types_t>;

    static constexpr std::size_t n_layers = sizeof...(Layers);

    template <std::size_t I, class T>
    struct layer_leaf {
        [[no_unique_address]] T value;

        constexpr layer_leaf() = default;

        template <class U>
        constexpr explicit layer_leaf(U&& v) : value(std::forward<U>(v)) {}
    };

    template <class Seq, class... Ts>
    struct layers_storage_impl;

    template <std::size_t... Is, class... Ts>
    struct layers_storage_impl<std::index_sequence<Is...>, Ts...>
        : layer_leaf<Is, Ts>... {
        constexpr layers_storage_impl() = default;

        template <class... Us>
        constexpr explicit layers_storage_impl(Us&&... us)
            : layer_leaf<Is, Ts>(std::forward<Us>(us))... {}
    };

    using layers_storage_t =
        layers_storage_impl<std::make_index_sequence<n_layers>, Layers...>;

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
        using in_t = ::layer_in_t<L>;
        using out_t = ::layer_out_t<L>;

        static_assert(
            StaticTensorView<in_t>,
            "Sequential layer input must satisfy StaticTensorView."
        );
        static_assert(
            StaticTensorView<out_t>,
            "Sequential layer output must satisfy StaticTensorView."
        );
        static_assert(
            !std::is_const_v<typename out_t::value_type>,
            "Sequential layer output tensor value_type must be non-const."
        );
    }

    template <class ALayer, class BLayer>
    static consteval bool adjacent_match() {
        using a_out = ::layer_out_t<ALayer>;
        using b_in = ::layer_in_t<BLayer>;

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

   public:
    using allocator_type = A;
    using input_tensor_type = ::layer_in_t<layer_t<0>>;
    using output_tensor_type = ::layer_out_t<layer_t<n_layers - 1>>;

    static constexpr std::size_t scratch_bytes = [] {
        if constexpr (n_layers <= 1) return std::size_t{0};
        std::size_t max_bytes = 0;
        [&]<std::size_t... Is>(std::index_sequence<Is...>) constexpr {
            ((max_bytes = std::max(max_bytes, ::layer_out_t<layer_t<Is>>::byte_size)), ...);
        }(std::make_index_sequence<n_layers - 1>{});  // intermediate outputs: [0..n-2]
        return max_bytes;
    }();

    static constexpr std::size_t scratch_alignment = [] {
        if constexpr (n_layers <= 1) return std::size_t{1};
        std::size_t max_align = 1;
        [&]<std::size_t... Is>(std::index_sequence<Is...>) constexpr {
            ((max_align = std::max(max_align, ::layer_out_t<layer_t<Is>>::alignment)),
             ...);
        }(std::make_index_sequence<n_layers - 1>{});  // intermediate outputs: [0..n-2]
        return max_align;
    }();

   private:
    [[no_unique_address]] A allocator_;
    [[no_unique_address]] layers_storage_t layers_;

    template <std::size_t I>
    constexpr decltype(auto) layer() noexcept {
        using leaf_t = layer_leaf<I, layer_t<I>>;
        return static_cast<leaf_t&>(layers_).value;
    }

    template <std::size_t I>
    constexpr decltype(auto) layer() const noexcept {
        using leaf_t = layer_leaf<I, layer_t<I>>;
        return static_cast<const leaf_t&>(layers_).value;
    }

    template <std::size_t I>
    using in_t = ::layer_in_t<layer_t<I>>;

    template <std::size_t I>
    using out_t = ::layer_out_t<layer_t<I>>;

    template <std::size_t I>
    constexpr void run_stage(std::byte* in_buf, std::byte* out_buf) {
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

    constexpr A& allocator() noexcept { return allocator_; }
    constexpr const A& allocator() const noexcept { return allocator_; }

    constexpr void operator()(const input_tensor_type& in, output_tensor_type& out) {
        if constexpr (n_layers == 1) {
            layer<0>()(in, out);
            return;
        }

        // Allocate two aligned scratch buffers big enough for the largest
        // intermediate tensor. We over-allocate and then bump the returned
        // pointer to `scratch_alignment`.
        struct scratch_deleter {
            A* alloc{};
            std::size_t bytes{};

            void operator()(std::byte* p) const noexcept {
                if (p == nullptr) return;
                alloc->deallocate(p, bytes);
            }
        };
        using scratch_ptr = std::unique_ptr<std::byte, scratch_deleter>;

        struct scratch_block {
            scratch_ptr raw;
            std::byte* aligned{};
        };

        auto make_scratch = [&]() -> scratch_block {
            if (scratch_bytes == 0) return scratch_block{scratch_ptr(nullptr, scratch_deleter{&allocator_, 0}), nullptr};

            const std::size_t raw_bytes = scratch_bytes + (scratch_alignment - 1);
            scratch_ptr raw(
                allocator_.allocate(raw_bytes),
                scratch_deleter{&allocator_, raw_bytes}
            );

            void* p = raw.get();
            std::size_t space = raw_bytes;
            void* ap = std::align(scratch_alignment, scratch_bytes, p, space);
            return scratch_block{std::move(raw), static_cast<std::byte*>(ap)};
        };

        scratch_block ping = make_scratch();
        scratch_block pong = make_scratch();

        // Stage 0: in -> ping
        {
            auto out_view = make_tensor_view<out_t<0>>(ping.aligned);
            layer<0>()(in, out_view);
        }

        // Middle stages: ping <-> pong
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            // Is maps to stage index (1..n_layers-2)
            (([&] {
                 constexpr std::size_t stage = Is + 1;
                 run_stage<stage>(ping.aligned, pong.aligned);
                 std::swap(ping, pong);
             }()),
             ...);
        }(std::make_index_sequence<n_layers - 2>{});

        // Final stage: ping -> out
        {
            auto in_view = make_tensor_view<in_t<n_layers - 1>>(ping.aligned);
            layer<n_layers - 1>()(in_view, out);
        }
    }
};

#endif
