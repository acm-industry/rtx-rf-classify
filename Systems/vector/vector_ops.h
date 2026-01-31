#ifndef VEC_OPS_H
#define VEC_OPS_H

#include <cstddef>

// Compile with -DUSE_HIGHWAY to use the Highway SIMD backend.
// Without it, a sequential implementation will be used.

namespace vec {

void add(const float* a, const float* b, float* out, size_t n);
void dot(const float* a, const float* b, float* out, size_t n);

}

#endif
