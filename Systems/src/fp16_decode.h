#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

namespace rtx::fp16 {

/**
 * IEEE 754 binary16 to binary32. Used to expand FP16 weight blobs to float for inference.
 */
inline float half_to_float(uint16_t h) {
    const unsigned s = h >> 15;
    const unsigned e = (h >> 10) & 31u;
    const unsigned m = h & 1023u;
    const float sign = s ? -1.f : 1.f;

    if (e == 0u) {
        if (m == 0u) {
            return sign * 0.f;
        }
        // Subnormal: m * 2^-24
        return sign * std::ldexp(static_cast<float>(m), -24);
    }
    if (e == 31u) {
        if (m == 0u) {
            return sign * std::numeric_limits<float>::infinity();
        }
        return std::numeric_limits<float>::quiet_NaN();
    }
    // Normal: (1 + m/1024) * 2^(e - 15)
    return sign * std::ldexp(1.f + static_cast<float>(m) / 1024.f, static_cast<int>(e) - 15);
}

} // namespace rtx::fp16
