#!/usr/bin/env bash
set -euo pipefail

# Expected location:
#   <ara-root>/apps/rtx-rf-classify/Systems/ara_run.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ARA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd -P)"
ARA_APPS_DIR="${ARA_ROOT}/apps"

APP="${APP:-rf_classify}"
RFIO_BIND="${RFIO_BIND:-127.0.0.1}"
RFIO_PORT="${RFIO_PORT:-9090}"
CROSS_PREFIX="${CROSS_PREFIX:-}"
READELF="${READELF:-}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

find_readelf() {
  if [[ -n "${READELF}" ]]; then
    command_exists "${READELF}" || die "READELF not found: ${READELF}"
    return
  fi

  if [[ -n "${CROSS_PREFIX}" && -x "${CROSS_PREFIX}readelf" ]]; then
    READELF="${CROSS_PREFIX}readelf"
  elif command_exists riscv64-linux-gnu-readelf; then
    READELF="riscv64-linux-gnu-readelf"
  elif command_exists riscv64-unknown-elf-readelf; then
    READELF="riscv64-unknown-elf-readelf"
  elif command_exists readelf; then
    READELF="readelf"
  fi
}

check_load_addresses() {
  local app_bin="$1"
  local bad_load_addr=0
  local vaddr

  find_readelf
  if [[ -z "${READELF}" ]]; then
    echo "WARNING: no readelf found; skipping ELF load-address sanity check" >&2
    return
  fi

  while read -r vaddr; do
    if (( vaddr < 0x80000000 )); then
      bad_load_addr=1
    fi
  done < <("${READELF}" -lW "${app_bin}" | awk '$1 == "LOAD" { print $3 }')

  if [[ "${bad_load_addr}" == "1" ]]; then
    "${READELF}" -lW "${app_bin}" >&2
    die "ELF has a PT_LOAD segment below 0x80000000; rebuild with ./ara_build.sh"
  fi
}

[[ -d "${ARA_ROOT}/hardware" ]] || die "missing Ara hardware directory: ${ARA_ROOT}/hardware"
[[ -d "${ARA_APPS_DIR}" ]] || die "missing Ara apps directory: ${ARA_APPS_DIR}"
[[ -x "${ARA_ROOT}/run.sh" ]] || die "missing executable Ara run script: ${ARA_ROOT}/run.sh"
[[ -f "${ARA_APPS_DIR}/bin/${APP}" ]] || die "missing app binary: ${ARA_APPS_DIR}/bin/${APP}; run ./ara_build.sh first"

check_load_addresses "${ARA_APPS_DIR}/bin/${APP}"

echo "Running ${APP}"
echo "RFIO: ${RFIO_BIND}:${RFIO_PORT}"

cd "${ARA_ROOT}"
BUILD_APP=0 APP="${APP}" RFIO_BIND="${RFIO_BIND}" RFIO_PORT="${RFIO_PORT}" ./run.sh
