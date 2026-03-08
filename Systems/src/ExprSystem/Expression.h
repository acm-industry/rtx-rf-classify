#ifndef __EXPRESSION_H__
#define __EXPRESSION_H__

#include <cstddef>
#include <concepts>
#include <tuple>

template <class E>
concept Expression = requires( const E& e, std::size_t idx ) {
    { E::extents };
    { E::iter_size() } -> std::same_as<size_t>;
    typename E::value_type;
    { e.access( idx ) } -> std::convertible_to< typename E::value_type >;
};

template <class Callable, class OperandType, std::size_t N_args>
concept Operation =
    std::is_empty_v<Callable> &&
    std::is_default_constructible_v<Callable> &&
    requires ( Callable callable, std::array<OperandType, N_args> arguments ) {
        { std::apply( callable, arguments ) } -> std::same_as<OperandType>;
    };




template <class Op, Expression... Es> requires (
    sizeof...(Es) > 0 && // More than one argument
    (std::convertible_to<typename Es::value_type, typename std::tuple_element<0, std::tuple<Es...>>::type::value_type> && ...) && // All have the same OperandType
    ( (Es::iter_size() == std::tuple_element<0, std::tuple<Es...>>::type::iter_size() ) && ... ) && // All have the same iteration size 
    Operation<Op, typename std::tuple_element<0, std::tuple<Es...>>::type::value_type, sizeof...(Es)> // Verify Op is an Operation
)
class OpExpr {
    std::tuple<Es...> children;

public:
    using tuple_elem_zero = typename std::tuple_element_t<0, std::tuple<Es...>>;

    using value_type = typename tuple_elem_zero::value_type;
    static constexpr auto extents = tuple_elem_zero::extents;
    static constexpr size_t iter_size() { return tuple_elem_zero::iter_size(); }

    constexpr explicit OpExpr(Es... es) : children{std::move(es)...} {}

    constexpr value_type access( auto... idx ) const { 
        return std::apply( 
            [&](auto const&... e) { return Op{}( e.access(idx...)... ); }, 
            children 
        ); 
    }
    
    class iterator {
        OpExpr const* owner;
        size_t i;

    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = OpExpr::value_type;
        using difference_type = std::ptrdiff_t;
        using reference = value_type; 
        using pointer = void;

        constexpr iterator(OpExpr const* o = nullptr, size_t idx = 0) : owner(o), i(idx) {}

        constexpr reference operator*() const {
            return owner->access(i);
        }

        constexpr iterator& operator++() {
            ++i;
            return *this;
        }

        constexpr iterator operator++(int) {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        friend constexpr bool operator==(const iterator& a, const iterator& b) {
            return a.owner == b.owner && a.i == b.i;
        }

        friend constexpr bool operator!=(const iterator& a, const iterator& b) {
            return !(a == b);
        }
    };

    constexpr iterator begin() const { return {this, 0}; }
    constexpr iterator end() const { return {this, iter_size()}; }

};

template<class Extents> 
constexpr std::size_t total_size(Extents) { 
    std::size_t size = 1; 
    for (std::size_t i = 0; i < Extents::rank(); ++i) size *= Extents::static_extent(i);
    return size;
}

template <Expression E, Expression Out> requires (
    std::convertible_to<typename E::value_type, typename Out::value_type> &&
    E::extents == Out::extents 
)
constexpr void in_place_eval( const E& expr, Out& out ) {
    for (size_t i = 0; i < E::iter_size(); ++i) out.access(i) = expr.access(i);
}

#endif
