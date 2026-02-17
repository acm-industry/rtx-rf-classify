#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

CXX_BIN="${CXX:-c++}"
COMMON_FLAGS=(-std=c++23 -O3 -DNDEBUG -march=native -I"${ROOT_DIR}")

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

RUNTIME_ASM="${TMP_DIR}/tensor_expr_runtime.s"
CONSTD_ASM="${TMP_DIR}/tensor_expr_constdata.s"
RUNTIME_KERNEL_SRC="${TMP_DIR}/tensor_expr_codegen_runtime.cpp"
CONSTD_KERNEL_SRC="${TMP_DIR}/tensor_expr_codegen_constdata.cpp"

cat > "${RUNTIME_KERNEL_SRC}" <<'CPP'
#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <cstdio>
#include <utility>

#include "Systems/src/ExprSystem/ExprFunctions.h"
#include "Systems/src/tensor.h"

using Vec64 = TensorBase<float, std::extents<size_t, 64>>;

float eval_kernel(const Vec64& a, const Vec64& b, const Vec64& c, Vec64& out) {
    in_place_eval(a + b * c, out);
    float sum = 0.0f;
    for (size_t i = 0; i < Vec64::iter_size(); ++i) sum += out.access(i);
    return sum;
}

int main() {
    std::array<float, Vec64::iter_size()> a_data{};
    std::array<float, Vec64::iter_size()> b_data{};
    std::array<float, Vec64::iter_size()> c_data{};

    unsigned seed = 12345u;
    for (size_t i = 0; i < Vec64::iter_size(); ++i) {
        seed = seed * 1664525u + 1013904223u;
        a_data[i] = static_cast<float>(seed & 0x3ffu) - 500.0f;
        seed = seed * 1664525u + 1013904223u;
        b_data[i] = static_cast<float>(seed & 0x3ffu) * 0.125f;
        seed = seed * 1664525u + 1013904223u;
        c_data[i] = static_cast<float>(seed & 0x1ffu) * 0.25f;
    }

    const Vec64 a(a_data.data());
    const Vec64 b(b_data.data());
    const Vec64 c(c_data.data());
    Vec64 out;

    float checksum = 0.0f;
    for (int rep = 0; rep < 1000; ++rep) checksum += eval_kernel(a, b, c, out);
    std::printf("%f\n", checksum);
    return 0;
}
CPP

cat > "${CONSTD_KERNEL_SRC}" <<'CPP'
#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <cstdio>
#include <utility>

#include "Systems/src/ExprSystem/ExprFunctions.h"
#include "Systems/src/tensor.h"

using Vec16 = TensorBase<float, std::extents<size_t, 16>>;

int main() {
    constexpr std::array<float, 16> left = {0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,
                                            6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f,
                                            12.0f, 13.0f, 14.0f, 15.0f};
    constexpr std::array<float, 16> right = {10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f,
                                             10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f,
                                             10.0f, 10.0f, 10.0f, 10.0f};

    const Vec16 a(left.data());
    const Vec16 b(right.data());
    Vec16 out;

    in_place_eval(a + b, out);

    float checksum = 0.0f;
    for (size_t i = 0; i < Vec16::iter_size(); ++i) checksum += out.access(i);
    std::printf("checksum=%f\n", checksum);
    return 0;
}
CPP

echo "compiling runtime-data expression test to assembly..."
"${CXX_BIN}" "${COMMON_FLAGS[@]}" -S "${RUNTIME_KERNEL_SRC}" -o "${RUNTIME_ASM}"

echo "compiling const-data expression test to assembly..."
"${CXX_BIN}" -std=c++23 -O3 -march=native -I"${ROOT_DIR}" -S "${CONSTD_KERNEL_SRC}" -o "${CONSTD_ASM}"

ARCH="$(uname -m)"
if [[ "${ARCH}" == "arm64" || "${ARCH}" == "aarch64" ]]; then
  SIMD_REGEX='fadd\.4s|fmul\.4s|fmla\.4s|faddp|fmadd'
else
  SIMD_REGEX='vaddps|vmulps|addps|mulps'
fi

echo "checking for SIMD instructions (${ARCH})..."
if ! grep -Eq "${SIMD_REGEX}" "${RUNTIME_ASM}"; then
  echo "[FAIL] no expected SIMD instructions found in ${RUNTIME_ASM}" >&2
  exit 1
fi

echo "checking const-data loading patterns..."
LEFT_REGEX='const\.main\.left|__const\.main\.left|left@PAGE|left\(%rip\)'
RIGHT_REGEX='const\.main\.right|__const\.main\.right|right@PAGE|right\(%rip\)'
if ! grep -Eq "${LEFT_REGEX}" "${CONSTD_ASM}" || ! grep -Eq "${RIGHT_REGEX}" "${CONSTD_ASM}"; then
  echo "[FAIL] const-data labels for left/right were not found in ${CONSTD_ASM}" >&2
  exit 1
fi

echo "[PASS] compiler optimization checks found SIMD ops and const-data loading markers."
