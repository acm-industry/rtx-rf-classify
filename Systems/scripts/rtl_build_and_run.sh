#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IMAGE_NAME="systems-riscv-rtl:latest"

# ── Parse arguments ──
TRACE=""
ARA_CONFIG="default"
APP="kernels"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --trace)    TRACE="1"; shift ;;
        --config)   ARA_CONFIG="$2"; shift 2 ;;
        --config=*) ARA_CONFIG="${1#*=}"; shift ;;
        --app)      APP="$2"; shift 2 ;;
        --app=*)    APP="${1#*=}"; shift ;;
        -h|--help)
            echo "Usage: $0 [--trace] [--config <2_lanes|4_lanes|8_lanes|16_lanes>] [--app <kernels|mvp>]"
            echo ""
            echo "  --trace       Generate FST waveform traces (slower, larger output)"
            echo "  --config CFG  Ara lane configuration (default: default = 4 lanes)"
            echo "  --app APP     Which app to build & simulate:"
            echo "                  kernels (default) - test_rvv_kernels_baremetal.c"
            echo "                  mvp               - full classification MVP via"
            echo "                                      Systems/apps/classify_mvp/"
            echo ""
            echo "First run builds the Docker image (~2-4 hours for LLVM + Verilator + hardware)."
            echo "Subsequent runs use the cached image."
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

case "${APP}" in
    kernels|mvp) ;;
    *) echo "--app must be one of: kernels, mvp (got: ${APP})" >&2; exit 1 ;;
esac

echo "============================================"
echo "  Ara RTL Simulation Pipeline"
echo "  App: ${APP}  Config: ${ARA_CONFIG}  Trace: ${TRACE:-off}"
echo "============================================"
echo ""

# ── Build Docker image ──
echo "Building Docker image (first run takes 2-4 hours for LLVM/Verilator/hardware)..."
docker build -f "${ROOT_DIR}/docker/Dockerfile.riscv-rtl" -t "${IMAGE_NAME}" "${ROOT_DIR}"

# ── Run verification inside the container ──
docker run --rm \
    -v "${ROOT_DIR}:/work" \
    -w /work \
    -e "ARA_CONFIG=${ARA_CONFIG}" \
    -e "TRACE=${TRACE}" \
    -e "APP=${APP}" \
    "${IMAGE_NAME}" \
    bash -lc "./scripts/verify_rtl_simulation.sh"
