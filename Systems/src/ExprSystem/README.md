# ExprSystem

## Purpose

This directory holds the Expression system library. This is key to using lazy-evaluation to maximize compile-time advantages with C++'s type system. 

Effectively, we leverage C++'s template system to "keep track" of the AST of our operations is we go along. For example,
```cpp
constexpr Vector<16> left{data1}; constexpr Vector<16> right{data2};
auto expression = left + right * right;
```
As of right now, left and right are dimension-16 vectors holding data. The type of `expression` is going to be something like `Add< Vector<16>, Multiply< Vector<16>, Vector<16> > >`, because nothing is added here yet. `expression` has no data that you can print, other than references to the left and right vectors. But, later,
```cpp
Vector<16> out{buffer};
in_place_eval(expression, out);
```
This will run a single for loop to compute `left + right * right`. The resulting assembly is something like:
`for (size_t i = 0; i < 16; ++i) out[i] = left[i] + right[i] * right[i];`
If we immediately evaluted,
```cpp
for (size_t i = 0; i < 16; ++i) out[i] = right[i] * right[i];
for (size_t i = 0; i < 16; ++i) out[i] = left[i] + out[i];
```
Notice the number of loops over the exact same data regions (`out, left, right` buffers) grows linearly with our number of operations if we evaluate immedatiely. If we have `N` operations, we retread the same memory `N` times. For the most part, memory operations are far slower than CPU computation, especially on constrained devices. Using lazy evaluation, we perform the compute with `O(1)` retreads over the same memory. This is already a major performance boost.

Furthermore, we gain performance due to "simple loops". The compiler is great at optimizing these. A good example of just how much benefit we get is in /Systems/tests/test_expr*.cpp. `constdata` demonstrates how the compiler does all of the compute itself on adding the vectors, and `runtime` shows how the compiler optimizes the addition into AVX2 simd instructions, with 8-byte wide registers.

Using the expression system means we get the ease of use of normal syntax, with the benefit of major comp-time optimizations.

## Files

- `Broadcast.h`: This expands upon the default `std::extents` to allow for compile-time numpy-style broadcasting. It provides:
1. `broadcast_extents<E1, E2, ...>` which uses Numpy broadcasting rules to generate a `std::extents`. For example, `broadcast_extents< std::extents<size_t, 2,2>, std::extents<size_t, 2> >` is equal to the type `std::extents<size_t, 2, 2>`. This will be usable in the future if we want to include this kind of broadcasting with the expression system.

- `Expression.h`: This includes the heart of the expression system. It provides:
1. `concept Expression<E>`: requires `E::extents` (must be a `std::extents`), `E::iter_size()` (defines a single-loop "size" to iterate through; this can be the number of items, number of vectors, etc. to be used in tandem with `access`), `typename E::value_type` (does not have to be an actual value, but it should be valid to run the operations you run on it, like `exp(value_type{})`), `e.access(size_t)` (this should "access" one item. This can be a vector, a single float, a SIMD vector, anything that can work with the operations you run on it. It should return a `value_type`). **All** values we want to work with the expression should satisfy this concept to be used. Check `test` for an example with a basic `Vector` type. Use `constexpr bool test = Expression<type>;` to verify.
2. `in_place_eval( const Expression& expr, Expression& writable_expression )`: The `writable_expression` type's `access(size_t)` method should be assignable or have an assignable, i.e., `writable_expression.access(i) = expr.access(i)` should be a valid expression. We prefer to use `in_place` to avoid unseen allocations.

- `ExprFunctions.h`: This has a bunch of functions immediately usable with the expression system. The functions available are:
1. Basic arithmetic, `a + b, a - b, a / b, a * b, pow(a, b), -a`. If you have a type that satisfies `concept Expression`, the compiler will automatically know to apply the operator overloads for arithmetic. This you can do `a + b * c - a / (-b)`, which will create the expression tree without evaluation.
2. Some basic math functions, `sin, cos, tan, tanh, exp, log, relu`.
If you want to make one yourself, define a functor that does the operation. For example,
```cpp
struct Add { template <class T> T operator()(T a, T b) { return a + b; } };
```
Then define a function that applies the expression:
```cpp
template <Expression E1, Expression E2>
constexpr auto add( E1 a, E2 b ) { return OpExpr<Add, E1, E2>{std::move(a), std::move(b)}; }
```
Then, you can call `add(expr1, expr2)` and it will work with the expression system.



