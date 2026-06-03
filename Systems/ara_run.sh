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

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -d "${ARA_ROOT}/hardware" ]] || die "missing Ara hardware directory: ${ARA_ROOT}/hardware"
[[ -d "${ARA_APPS_DIR}" ]] || die "missing Ara apps directory: ${ARA_APPS_DIR}"
[[ -x "${ARA_ROOT}/run.sh" ]] || die "missing executable Ara run script: ${ARA_ROOT}/run.sh"
[[ -f "${ARA_APPS_DIR}/bin/${APP}" ]] || die "missing app binary: ${ARA_APPS_DIR}/bin/${APP}; run ./ara_build.sh first"

echo "Running ${APP}"
echo "RFIO: ${RFIO_BIND}:${RFIO_PORT}"

cd "${ARA_ROOT}"
BUILD_APP=0 APP="${APP}" RFIO_BIND="${RFIO_BIND}" RFIO_PORT="${RFIO_PORT}" ./run.sh
