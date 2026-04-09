#ifndef __SCALAR_EXPR_H__
#define __SCALAR_EXPR_H__

#include <cstdint>

#include "Expression.h"
#include "Broadcast.h"

template <class OperandType>
struct Scalar {
    static constexpr auto extents = std::extents<size_t>();
    using value_type = OperandType;

    static constexpr size_t iter_size() { return 1; }

    OperandType value;

    constexpr Scalar( OperandType val ) : value(val) {}
    constexpr Scalar() = default;

    constexpr const value_type& access(size_t...) const noexcept {
        return value;
    }
    
    constexpr value_type& access(size_t...) noexcept {
        return value;
    }

    constexpr operator OperandType() const {
        return value;
    }

    template <class TargetExtents>
    constexpr operator BroadcastExpr<Scalar<OperandType>, TargetExtents>() const {
        return BroadcastExpr<Scalar<OperandType>, TargetExtents>{*this};
    }
};

template <class TargetExtents, class T>
constexpr auto broadcast(T value) {
    using scalar_type = Scalar<decltype(value)>;
    return BroadcastExpr<scalar_type, TargetExtents>{scalar_type(value)};
}

constexpr Scalar<std::uint64_t> operator"" _sui64(unsigned long long v) {
    return Scalar<std::uint64_t>{static_cast<std::uint64_t>(v)};
}

constexpr Scalar<std::uint32_t> operator"" _sui32(unsigned long long v) {
    return Scalar<std::uint32_t>{static_cast<std::uint32_t>(v)};
}

constexpr Scalar<std::uint16_t> operator"" _sui16(unsigned long long v) {
    return Scalar<std::uint16_t>{static_cast<std::uint16_t>(v)};
}

constexpr Scalar<std::uint8_t> operator"" _sui8(unsigned long long v) {
    return Scalar<std::uint8_t>{static_cast<std::uint8_t>(v)};
}

constexpr Scalar<std::int64_t> operator"" _si64(unsigned long long v) {
    return Scalar<std::int64_t>{static_cast<std::int64_t>(v)};
}

constexpr Scalar<std::int32_t> operator"" _si32(unsigned long long v) {
    return Scalar<std::int32_t>{static_cast<std::int32_t>(v)};
}

constexpr Scalar<std::int16_t> operator"" _si16(unsigned long long v) {
    return Scalar<std::int16_t>{static_cast<std::int16_t>(v)};
}

constexpr Scalar<std::int8_t> operator"" _si8(unsigned long long v) {
    return Scalar<std::int8_t>{static_cast<std::int8_t>(v)};
}

constexpr Scalar<bool> operator"" _sb(unsigned long long v) {
    return Scalar<bool>{v != 0ULL};
}

constexpr Scalar<float> operator"" _sf(long double v) {
    return Scalar<float>{static_cast<float>(v)};
}

constexpr Scalar<double> operator"" _sd(long double v) {
    return Scalar<double>{static_cast<double>(v)};
}

constexpr Scalar<long double> operator"" _sld(long double v) {
    return Scalar<long double>{v};
}




#endif
