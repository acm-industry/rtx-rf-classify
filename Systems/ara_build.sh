#!/usr/bin/env bash
set -euo pipefail

# Expected location:
#   <ara-root>/apps/rtx-rf-classify/Systems/ara_build.sh
#
# This script keeps the Git checkout intact and generates a small Ara app at:
#   <ara-root>/apps/rf_classify

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
ARA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd -P)"
ARA_APPS_DIR="${ARA_ROOT}/apps"
COMMON_DIR="${ARA_APPS_DIR}/common"

APP="${APP:-rf_classify}"
SRC_DIR="${SRC_DIR:-${SCRIPT_DIR}/src}"
BIN_DIR="${BIN_DIR:-${SCRIPT_DIR}/scripts/binaries}"
APP_DIR="${ARA_APPS_DIR}/${APP}"
MDSPAN_INCLUDE_DIR="${MDSPAN_INCLUDE_DIR:-}"
MDSPAN_FETCH_DIR="${MDSPAN_FETCH_DIR:-${REPO_ROOT}/build/_deps/mdspan-src}"
MDSPAN_GIT_URL="${MDSPAN_GIT_URL:-https://github.com/kokkos/mdspan.git}"
MDSPAN_GIT_REF="${MDSPAN_GIT_REF:-stable}"
CROSS_PREFIX="${CROSS_PREFIX:-}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "missing required file: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "missing required directory: $1"
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

require_dir "${ARA_APPS_DIR}"
require_dir "${COMMON_DIR}"
require_dir "${SRC_DIR}"
require_dir "${SRC_DIR}/ExprSystem"
require_dir "${BIN_DIR}"
require_file "${SRC_DIR}/main.cpp"
require_file "${COMMON_DIR}/rfio.h"
require_file "${COMMON_DIR}/link.ld"

case "${APP}" in
  ""|common|bin|riscv-tests|rtx-rf-classify|.*|*/*)
    die "unsafe app name: ${APP}"
    ;;
esac

[[ "${APP_DIR}" != "${SCRIPT_DIR}" ]] || die "refusing to overwrite this checkout"
if [[ -d "${APP_DIR}/.git" ]]; then
  die "refusing to overwrite git checkout: ${APP_DIR}"
fi

find_mdspan() {
  local candidate

  if [[ -n "${MDSPAN_INCLUDE_DIR}" ]]; then
    require_dir "${MDSPAN_INCLUDE_DIR}/experimental"
    require_dir "${MDSPAN_INCLUDE_DIR}/mdspan"
    return
  fi

  mdspan_candidates=(
    "${REPO_ROOT}/build/_deps/mdspan-src/include"
    "${SCRIPT_DIR}/../build/_deps/mdspan-src/include"
    "${SCRIPT_DIR}/build/_deps/mdspan-src/include"
    "${ARA_ROOT}/build/_deps/mdspan-src/include"
    "${ARA_ROOT}/../build/_deps/mdspan-src/include"
    "${MDSPAN_FETCH_DIR}/include"
  )

  for candidate in "${mdspan_candidates[@]}"; do
    if [[ -d "${candidate}/experimental" && -d "${candidate}/mdspan" ]]; then
      MDSPAN_INCLUDE_DIR="${candidate}"
      return
    fi
  done
}

fetch_mdspan() {
  echo "mdspan:        fetching ${MDSPAN_GIT_URL} (${MDSPAN_GIT_REF})"
  mkdir -p "$(dirname "${MDSPAN_FETCH_DIR}")"

  if [[ -d "${MDSPAN_FETCH_DIR}/.git" ]]; then
    if git -C "${MDSPAN_FETCH_DIR}" fetch --depth 1 origin "${MDSPAN_GIT_REF}"; then
      git -C "${MDSPAN_FETCH_DIR}" checkout -q FETCH_HEAD
    else
      git -C "${MDSPAN_FETCH_DIR}" fetch --depth 1 origin
    fi
  else
    if ! git clone --depth 1 --branch "${MDSPAN_GIT_REF}" "${MDSPAN_GIT_URL}" "${MDSPAN_FETCH_DIR}"; then
      rm -rf "${MDSPAN_FETCH_DIR}"
      git clone --depth 1 "${MDSPAN_GIT_URL}" "${MDSPAN_FETCH_DIR}"
      git -C "${MDSPAN_FETCH_DIR}" checkout -q "${MDSPAN_GIT_REF}" 2>/dev/null || true
    fi
  fi

  MDSPAN_INCLUDE_DIR="${MDSPAN_FETCH_DIR}/include"
  require_dir "${MDSPAN_INCLUDE_DIR}/experimental"
  require_dir "${MDSPAN_INCLUDE_DIR}/mdspan"
}

find_mdspan
if [[ -z "${MDSPAN_INCLUDE_DIR}" ]]; then
  fetch_mdspan
fi

find_cross_prefix() {
  local prefix

  if [[ -n "${CROSS_PREFIX}" ]]; then
    command_exists "${CROSS_PREFIX}g++" || die "CROSS_PREFIX does not provide g++: ${CROSS_PREFIX}g++"
    return
  fi

  cross_prefix_candidates=(
    "${ARA_ROOT}/install/riscv-gcc/bin/riscv64-unknown-elf-"
    "${ARA_ROOT}/cheshire/sw/cva6-sdk/buildroot/output/host/bin/riscv64-buildroot-linux-gnu-"
    "riscv64-linux-gnu-"
    "riscv64-unknown-elf-"
    "riscv64-buildroot-linux-gnu-"
  )

  for prefix in "${cross_prefix_candidates[@]}"; do
    if command_exists "${prefix}g++" && command_exists "${prefix}gcc"; then
      CROSS_PREFIX="${prefix}"
      return
    fi
  done

  die "no RISC-V GNU C++ toolchain found; set CROSS_PREFIX, e.g. CROSS_PREFIX=riscv64-linux-gnu-"
}

find_cross_prefix

mapfile -t weight_bins < <(find "${BIN_DIR}" -maxdepth 1 -type f -name '*.bin' | sort)
[[ "${#weight_bins[@]}" -gt 0 ]] || die "no .bin weight files found in ${BIN_DIR}"

echo "Ara root:      ${ARA_ROOT}"
echo "Source:        ${SRC_DIR}"
echo "Generated app: ${APP_DIR}"
echo "App name:      ${APP}"
echo "Toolchain:     ${CROSS_PREFIX}gcc / ${CROSS_PREFIX}g++"

rm -rf "${APP_DIR}"
mkdir -p "${APP_DIR}/ExprSystem" "${APP_DIR}/weights"

cp "${SRC_DIR}/main.cpp" "${APP_DIR}/main.cpp"

find "${SRC_DIR}" -maxdepth 1 -type f -name '*.h' -print0 |
  while IFS= read -r -d '' header; do
    case "$(basename "${header}")" in
      network.h)
        ;;
      *)
        cp "${header}" "${APP_DIR}/"
        ;;
    esac
  done

cp "${SRC_DIR}/ExprSystem/"*.h "${APP_DIR}/ExprSystem/"
cp "${COMMON_DIR}/rfio.h" "${APP_DIR}/rfio.h"

if [[ -f "${SRC_DIR}/manual_blas.h" ]]; then
  cp "${SRC_DIR}/manual_blas.h" "${APP_DIR}/manual_blas.h"
  cp "${SRC_DIR}/manual_blas.h" "${APP_DIR}/blas_ops.h"
fi

cp -R "${MDSPAN_INCLUDE_DIR}/experimental" "${APP_DIR}/experimental"
cp -R "${MDSPAN_INCLUDE_DIR}/mdspan" "${APP_DIR}/mdspan"
echo "mdspan:        ${MDSPAN_INCLUDE_DIR}"

for bin in "${weight_bins[@]}"; do
  cp "${bin}" "${APP_DIR}/weights/"
done

weights_s="${APP_DIR}/weights.S"
{
  echo '/* Generated by ara_build.sh. */'
  echo '.section .l2, "aw", @progbits'
  for bin in "${APP_DIR}/weights/"*.bin; do
    base="$(basename "${bin}" .bin)"
    abs="$(realpath "${bin}")"
    echo
    echo '.balign 32'
    echo ".global ${base}_start"
    echo ".global ${base}_end"
    echo ".global ${base}_size"
    echo "${base}_start:"
    echo "  .incbin \"${abs}\""
    echo "${base}_end:"
    echo ".set ${base}_size, ${base}_end - ${base}_start"
  done
} > "${weights_s}"

common_flags=(
  -march=rv64gcv
  -mabi=lp64d
  -mcmodel=medany
  -I"${APP_DIR}"
  -I"${COMMON_DIR}"
  -O3
  -ffast-math
  -fno-common
  -fno-builtin-printf
  -fno-stack-protector
  -ffunction-sections
  -fdata-sections
  "\$(DEFINES)"
)

c_flags=(
  "${common_flags[@]}"
  -std=gnu99
)

cxx_flags=(
  "${common_flags[@]}"
  -std=gnu++23
  -fno-exceptions
  -fno-rtti
  -fno-use-cxa-atexit
  -fno-threadsafe-statics
)

ld_flags=(
  -static
  -nostartfiles
  -nostdlib
  -lgcc
  -Wl,--gc-sections
  -T"${COMMON_DIR}/link.ld"
)

echo "Building ${APP}..."
make -B -C "${ARA_APPS_DIR}" "bin/${APP}" \
  COMPILER=gcc \
  "RISCV_CC=${CROSS_PREFIX}gcc" \
  "RISCV_CXX=${CROSS_PREFIX}g++" \
  "RISCV_OBJDUMP=${CROSS_PREFIX}objdump" \
  "RISCV_OBJDUMP_FLAGS=" \
  "RISCV_STRIP=${CROSS_PREFIX}strip" \
  "RISCV_CCFLAGS=${c_flags[*]}" \
  "RISCV_CXXFLAGS=${cxx_flags[*]}" \
  "RISCV_LDFLAGS=${ld_flags[*]}"

echo
echo "Built: ${ARA_APPS_DIR}/bin/${APP}"
