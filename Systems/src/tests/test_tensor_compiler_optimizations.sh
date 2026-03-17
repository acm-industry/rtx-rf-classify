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
RUNTIME_KERNEL_SRC="${SCRIPT_DIR}/test_tensor_expr_codegen_runtime.cpp"
CONSTD_KERNEL_SRC="${SCRIPT_DIR}/test_tensor_expr_codegen_constdata.cpp"

echo "compiling runtime-data expression test to assembly..."
"${CXX_BIN}" "${COMMON_FLAGS[@]}" -S "${RUNTIME_KERNEL_SRC}" -o "${RUNTIME_ASM}"

echo "compiling const-data expression test to assembly..."
"${CXX_BIN}" -std=c++23 -O3 -march=native -I"${ROOT_DIR}" -S "${CONSTD_KERNEL_SRC}" -o "${CONSTD_ASM}"

ARCH="$(uname -m)"
if [[ "${ARCH}" == "arm64" || "${ARCH}" == "aarch64" ]]; then
  SIMD_REGEX='fadd\.4s|fmul\.4s|fmla\.4s|faddp|fmadd'
elif [[ "${ARCH}" == "riscv64" ]]; then
  SIMD_REGEX='vsetvli|vsetivli|vle[0-9]+\.v|vse[0-9]+\.v|vfadd\.|vfmul\.|vfmadd\.|vfmacc\.'
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
