#ifndef __EXPRESSION_HPP__
#define __EXPRESSION_HPP__

#include <concepts>
#include <experimental/mdspan>
#include <tuple>
#include <cmath>

namespace rtxsys {

    template <class E>
    concept Expression = requires( const E& e, std::size_t idx ) {
        { E::extents };
        typename E::value_type;
        { e.flat( idx ) } -> std::convertible_to< typename E::value_type >;
    };

    template <class Callable, class OperandType, std::size_t N_args>
    concept Operation =
        std::is_empty_v<Callable> &&
        std::is_default_constructible_v<Callable> &&
        requires ( Callable callable, std::array<OperandType, N_args> arguments ) {
            { std::apply( callable, arguments ) } -> std::same_as<OperandType>;
        };

    template <class OperandType>
    struct ScalarExpr {
        static constexpr auto extents = std::extents<size_t>();
        using value_type = OperandType;

        OperandType value;

        constexpr ScalarExpr( OperandType val ) : value(val) {}
        constexpr ScalarExpr() = default;

        constexpr const value_type& flat(size_t...) const noexcept {
            return value;
        }
        
        constexpr value_type& flat(size_t...) noexcept {
            return value;
        }
    };


    template <class Op, Expression... Es> requires (
        sizeof...(Es) > 0 && // More than one argument
        (std::same_as<typename Es::value_type, typename std::tuple_element<0, std::tuple<Es...>>::type::value_type> && ...) && // All have the same OperandType
        ( (Es::extents == std::tuple_element<0, std::tuple<Es...>>::type::extents ) && ... ) && // All have the same extents / shape
        Operation<Op, typename std::tuple_element<0, std::tuple<Es...>>::type::value_type, sizeof...(Es)> // Verify Op is an Operation
    )
    class OpExpr {
        std::tuple<Es...> children;

    public:
        using value_type = typename std::tuple_element<0, std::tuple<Es...>>::type::value_type;

        static constexpr auto extents = std::tuple_element_t<0, std::tuple<Es...>>::extents;

        constexpr explicit OpExpr(Es... es) : children{std::move(es)...} {}

        constexpr value_type flat( auto... idx ) const { 
            return std::apply( 
                [&](auto const&... e) { return Op{}( e.flat(idx...)... ); }, 
                children 
            ); 
        }
        

    };

    struct Neg {
        template <class T>
        constexpr T operator()( T x ) const { return -x; }
    };

    struct Add {
        template <class T>
        constexpr T operator()( T a, T b ) const { return a + b; }
    };

    struct Sub {
        template <class T>
        constexpr T operator()( T a, T b ) const { return a - b; }
    };

    struct Mul {
        template <class T>
        constexpr T operator()( T a, T b ) const { return a * b; }
    };

    struct Div {
        template <class T>
        constexpr T operator()( T a, T b ) const { return a / b; }
    };

    template <Expression E>
    constexpr auto operator-(E e) {
        return OpExpr<Neg, E>{ std::move(e) };
    }

    template <Expression L, Expression R>
    constexpr auto operator+(L l, R r) {
        return OpExpr<Add, L, R>{ std::move(l), std::move(r) };
    }

    template <Expression L, Expression R>
    constexpr auto operator-(L l, R r) {
        return OpExpr<Sub, L, R>{ std::move(l), std::move(r) };
    }

    template <Expression L, Expression R>
    constexpr auto operator*(L l, R r) {
        return OpExpr<Mul, L, R>{ std::move(l), std::move(r) };
    }

    template <Expression L, Expression R>
    constexpr auto operator/(L l, R r) {
        return OpExpr<Div, L, R>{ std::move(l), std::move(r) };
    }

    template<class Extents> 
    constexpr std::size_t total_size(Extents) { 
        std::size_t size = 1; 
        for (std::size_t i = 0; i < Extents::rank(); ++i) size *= Extents::static_extent(i);
        return size;
    }

    template <class T> 
    concept WritableValue = requires( T& value, size_t i ) {
        typename T::value_type;
        { value.flat(i) } -> std::same_as<typename T::value_type&>;
    };

    template <Expression E, WritableValue Out> requires (
        std::same_as<typename E::value_type, typename Out::value_type> &&
        E::extents == Out::extents 
    )
    constexpr void in_place_eval( const E& expr, Out& out ) {
        constexpr size_t N = total_size( E::extents );

        for (size_t i = 0; i < N; ++i) out.flat(i) = expr.flat(i);
    }

    struct Sine {
        template <class T>
        constexpr T operator()( T x ) const { return std::sin(x); }
    };

    template <Expression E>
    constexpr auto sin(E e) {
        return OpExpr<Sine, E>{ std::move(e) };
    }


}

#endif