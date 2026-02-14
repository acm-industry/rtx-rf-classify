#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT_DIR}/vector_ops.cpp"
TEST="${ROOT_DIR}/tests/test_vector_ops.cpp"
CXX="${CXX:-g++}"
CXXFLAGS="-std=c++23 -O2 -Wall -Wextra -Wpedantic -fno-rtti"
HWY_INCLUDE="${HWY_INCLUDE:-/opt/homebrew/include}"
HWY_LIB="${HWY_LIB:-/opt/homebrew/lib}"

BUILD_DIR="${ROOT_DIR}/build/bench"
mkdir -p "${BUILD_DIR}"

OPENBLAS_INCLUDE="${OPENBLAS_INCLUDE:-/opt/homebrew/opt/openblas/include}"
OPENBLAS_LIB="${OPENBLAS_LIB:-/opt/homebrew/opt/openblas/lib}"

echo "============================================="
echo " Vector Ops Benchmark: Scalar vs Highway vs BLAS"
echo "============================================="
echo ""

# --- Scalar ---
echo "Compiling scalar backend..."
${CXX} ${CXXFLAGS} -I "${ROOT_DIR}" "${SRC}" "${TEST}" \
  -o "${BUILD_DIR}/test_vec_seq"

echo ""
echo "--- Scalar backend ---"
"${BUILD_DIR}/test_vec_seq"

# --- Highway ---
echo ""
echo "Compiling highway backend..."
${CXX} ${CXXFLAGS} -DUSE_HIGHWAY \
  -I "${ROOT_DIR}" -I "${HWY_INCLUDE}" \
  -L "${HWY_LIB}" -lhwy \
  -Wno-gnu-zero-variadic-macro-arguments \
  "${SRC}" "${TEST}" \
  -o "${BUILD_DIR}/test_vec_hwy"

echo ""
echo "--- Highway backend ---"
"${BUILD_DIR}/test_vec_hwy"

# --- BLAS (OpenBLAS) ---
echo ""
echo "Compiling BLAS backend..."
${CXX} ${CXXFLAGS} -DUSE_BLAS \
  -I "${ROOT_DIR}" -I "${OPENBLAS_INCLUDE}" \
  -L "${OPENBLAS_LIB}" -lopenblas \
  "${SRC}" "${TEST}" \
  -o "${BUILD_DIR}/test_vec_blas"

echo ""
echo "--- BLAS backend ---"
"${BUILD_DIR}/test_vec_blas"
