#pragma once

#include <cmath>
#include <concepts>
#include <cstddef>
#include <type_traits>

using fp16_t = _Float16;

template <class T>
inline constexpr bool is_supported_scalar_v =
    std::same_as<std::remove_cv_t<T>, fp16_t> ||
    std::same_as<std::remove_cv_t<T>, float> ||
    std::same_as<std::remove_cv_t<T>, double>;

template <class T>
concept SupportedScalar = is_supported_scalar_v<T>;

template <class T>
struct accumulation_type {
    using type = std::remove_cv_t<T>;
};

template <>
struct accumulation_type<fp16_t> {
    using type = float;
};

template <class T>
using accumulation_type_t = typename accumulation_type<std::remove_cv_t<T>>::type;

template <SupportedScalar T>
constexpr auto promote_for_math(T value) -> accumulation_type_t<T> {
    return static_cast<accumulation_type_t<T>>(value);
}

template <SupportedScalar T>
constexpr T cast_from_accum(accumulation_type_t<T> value) {
    return static_cast<T>(value);
}

template <SupportedScalar T>
inline T scalar_sqrt(T value) {
    using accum_t = accumulation_type_t<T>;
    return static_cast<T>(std::sqrt(static_cast<accum_t>(value)));
}

#if COMPUTE_FP16
using infer_t = fp16_t;
#else
using infer_t = float;
#endif
