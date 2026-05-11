#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IMAGE_NAME="systems-riscv-rtl:latest"
# Stable container name so a second terminal can `docker exec` into the
# running simulation while it's chugging through the Verilated chip model.
# Defaults to "ara-rtl" with the app suffix; override with --container-name.
DEFAULT_CONTAINER_NAME="ara-rtl"

# ── Parse arguments ──
TRACE=""
ARA_CONFIG="default"
APP="kernels"
SIMV_TIMEOUT="${SIMV_TIMEOUT:-1800}"   # 30 minutes default; 0 disables
CONTAINER_NAME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --trace)            TRACE="1"; shift ;;
        --config)           ARA_CONFIG="$2"; shift 2 ;;
        --config=*)         ARA_CONFIG="${1#*=}"; shift ;;
        --app)              APP="$2"; shift 2 ;;
        --app=*)            APP="${1#*=}"; shift ;;
        --simv-timeout)     SIMV_TIMEOUT="$2"; shift 2 ;;
        --simv-timeout=*)   SIMV_TIMEOUT="${1#*=}"; shift ;;
        --container-name)   CONTAINER_NAME="$2"; shift 2 ;;
        --container-name=*) CONTAINER_NAME="${1#*=}"; shift ;;
        -h|--help)
            cat <<EOF
Usage: $0 [--trace] [--config CFG] [--app <kernels|mvp>]
          [--simv-timeout SECS] [--container-name NAME]

  --trace               Generate FST waveform traces (slower, larger output)
  --config CFG          Ara lane configuration (default: default = 4 lanes)
                        Allowed: 2_lanes | 4_lanes | 8_lanes | 16_lanes
  --app APP             Which app to build & simulate:
                          kernels (default) - test_rvv_kernels_baremetal.c
                          mvp               - full classification MVP via
                                              Systems/apps/classify_mvp/
  --simv-timeout SECS   Hard timeout on the Verilator simv phase (default: 1800).
                        Pass 0 to disable.  Watchdog dumps the last 200 lines of
                        rtl_sim_output.txt and a 'ps' snapshot before killing simv.
  --container-name NAME Name to give the docker container so a second terminal
                        can attach with: docker exec -it <name> bash
                        (default: ${DEFAULT_CONTAINER_NAME}-<app>)

Diagnostics while a run is in progress (in another terminal):

  bash ${ROOT_DIR}/scripts/peek_rtl_sim.sh [container-name]

  - tails build/rtl_sim_output.txt
  - prints the most active processes inside the container (verilator/simv/make)
  - reports wall-clock + simv cycles if the runtime printed any

First run builds the Docker image (~2-4 hours for LLVM + Verilator + hardware).
Subsequent runs use the cached image.
EOF
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

case "${APP}" in
    kernels|mvp) ;;
    *) echo "--app must be one of: kernels, mvp (got: ${APP})" >&2; exit 1 ;;
esac

if [ -z "${CONTAINER_NAME}" ]; then
    CONTAINER_NAME="${DEFAULT_CONTAINER_NAME}-${APP}"
fi

# Helper to print a timestamped log line on the host side (the in-container
# script does its own timestamping; this is for the few host-side messages).
ts() { printf '\033[0;37m[%s]\033[0m ' "$(date '+%Y-%m-%d %H:%M:%S')"; }

echo "============================================"
echo "  Ara RTL Simulation Pipeline"
echo "  App:           ${APP}"
echo "  Config:        ${ARA_CONFIG}"
echo "  Trace:         ${TRACE:-off}"
echo "  Simv timeout:  ${SIMV_TIMEOUT}s$([ "${SIMV_TIMEOUT}" = 0 ] && echo ' (DISABLED)' || true)"
echo "  Container:     ${CONTAINER_NAME}"
echo "============================================"
echo ""
ts; echo "Starting pipeline.  To peek at the running simulation in another terminal:"
echo "    bash ${ROOT_DIR}/scripts/peek_rtl_sim.sh ${CONTAINER_NAME}"
echo ""

# ── Build Docker image (cached after the first run) ──
ts; echo "Step 1/2: Ensuring Docker image '${IMAGE_NAME}' exists (will use cache)..."
docker build -f "${ROOT_DIR}/docker/Dockerfile.riscv-rtl" -t "${IMAGE_NAME}" "${ROOT_DIR}"

# ── Clean up any stale container with the same name from a prior run ──
if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    ts; echo "Removing stale container '${CONTAINER_NAME}' from a previous run..."
    docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

# ── Run verification inside the container ──
ts; echo "Step 2/2: Launching verify_rtl_simulation.sh inside the container..."
echo ""

# -t allocates a pseudo-TTY so make/verilator don't block-buffer their stdout.
# --name lets the second terminal find the container easily.
# We deliberately use --rm so the container is cleaned up on exit; logs are
# preserved on the host via the bind-mounted ${ROOT_DIR} (rtl_sim_output.txt).
exec docker run --rm \
    --name "${CONTAINER_NAME}" \
    -t \
    -v "${ROOT_DIR}:/work" \
    -w /work \
    -e "ARA_CONFIG=${ARA_CONFIG}" \
    -e "TRACE=${TRACE}" \
    -e "APP=${APP}" \
    -e "SIMV_TIMEOUT=${SIMV_TIMEOUT}" \
    "${IMAGE_NAME}" \
    bash -lc "./scripts/verify_rtl_simulation.sh"
