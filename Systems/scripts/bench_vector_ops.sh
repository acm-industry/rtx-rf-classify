#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT_DIR}/src/vector/vector_ops.cpp"
TEST="${ROOT_DIR}/src/vector/tests/test_vector_ops.cpp"
CXX="${CXX:-g++}"
CXXFLAGS="-std=c++23 -O2 -Wall -Wextra -Wpedantic -fno-rtti"
HWY_INCLUDE="${HWY_INCLUDE:-/opt/homebrew/include}"
HWY_LIB="${HWY_LIB:-/opt/homebrew/lib}"

BUILD_DIR="${ROOT_DIR}/build/bench"
mkdir -p "${BUILD_DIR}"

echo "========================================"
echo " Vector Ops Benchmark: Scalar vs Highway"
echo "========================================"
echo ""

# --- Scalar ---
echo "Compiling scalar backend..."
${CXX} ${CXXFLAGS} -I "${ROOT_DIR}/src/vector" "${SRC}" "${TEST}" \
  -o "${BUILD_DIR}/test_vec_seq"

echo ""
echo "--- Scalar backend ---"
"${BUILD_DIR}/test_vec_seq"

# --- Highway ---
echo ""
echo "Compiling highway backend..."
${CXX} ${CXXFLAGS} -DUSE_HIGHWAY \
  -I "${ROOT_DIR}/src/vector" -I "${HWY_INCLUDE}" \
  -L "${HWY_LIB}" -lhwy \
  -Wno-gnu-zero-variadic-macro-arguments \
  "${SRC}" "${TEST}" \
  -o "${BUILD_DIR}/test_vec_hwy"

echo ""
echo "--- Highway backend ---"
"${BUILD_DIR}/test_vec_hwy"
