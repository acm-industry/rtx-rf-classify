#ifndef __EXPR_FUNCTIONS_H__
#define __EXPR_FUNCTIONS_H__

#include <algorithm>
#include <cmath>

#include "Expression.h"

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

struct Pow {
    template <class T>
    constexpr T operator()( T a, T b ) const { return std::pow(a, b); }
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
struct Sine {
    template <class T>
    constexpr T operator()( T x ) const { return std::sin(x); }
};

template <Expression E>
constexpr auto sin(E e) {
    return OpExpr<Sine, E>{ std::move(e) };
}

struct Cosine {
    template <class T>
    constexpr T operator()( T x ) const { return std::cos(x); }
};

template <Expression E>
constexpr auto cos(E e) {
    return OpExpr<Cosine, E>{ std::move(e) };
}

struct Tangent {
    template <class T>
    constexpr T operator()( T x ) const { return std::tan(x); }
};

template <Expression E>
constexpr auto tan(E e) {
    return OpExpr<Tangent, E>{ std::move(e) };
}

struct Exponential {
    template <class T>
    constexpr T operator()( T x ) const { return std::exp(x); }
};

template <Expression E>
constexpr auto exp(E e) {
    return OpExpr<Exponential, E>{ std::move(e) };
}

struct Logarithm {
    template <class T>
    constexpr T operator()( T x ) const { return std::log(x); }
};

template <Expression E>
constexpr auto log(E e) {
    return OpExpr<Logarithm, E>{ std::move(e) };
}

struct HyperbolicTangent {
    template <class T>
    constexpr T operator()( T x ) const { return std::tanh(x); }
};

template <Expression E>
constexpr auto tanh(E e) {
    return OpExpr<HyperbolicTangent, E>{ std::move(e) };
}

struct Sigmoid {
    template <class T>
    constexpr T operator()( T x ) const { return 1 / (1 + std::exp(-x)); }
};

template <Expression E>
constexpr auto sigmoid(E e) {
    return OpExpr<Sigmoid, E>{ std::move(e) };
}

struct ReLU {
    template <class T>
    constexpr T operator()( T x ) const { return std::max(T{0}, x); }
};

template <Expression E>
constexpr auto relu(E e) {
    return OpExpr<ReLU, E>{ std::move(e) };
}

struct Abs {
    template <class T>
    constexpr T operator()( T x ) const { return std::abs(x); }
};

template <Expression E>
constexpr auto abs(E e) {
    return OpExpr<Abs, E>{ std::move(e) };
}



#endif
