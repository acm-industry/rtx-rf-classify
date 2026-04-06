#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IMAGE_NAME="systems-riscv-rtl:latest"

# ── Parse arguments ──
TRACE=""
ARA_CONFIG="default"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --trace)    TRACE="1"; shift ;;
        --config)   ARA_CONFIG="$2"; shift 2 ;;
        --config=*) ARA_CONFIG="${1#*=}"; shift ;;
        -h|--help)
            echo "Usage: $0 [--trace] [--config <2_lanes|4_lanes|8_lanes|16_lanes>]"
            echo ""
            echo "  --trace       Generate FST waveform traces (slower, larger output)"
            echo "  --config CFG  Ara lane configuration (default: default = 4 lanes)"
            echo ""
            echo "First run builds the Docker image (~2-4 hours for LLVM + Verilator + hardware)."
            echo "Subsequent runs use the cached image."
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo "  Ara RTL Simulation Pipeline"
echo "  Config: ${ARA_CONFIG}  Trace: ${TRACE:-off}"
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
    "${IMAGE_NAME}" \
    bash -lc "./scripts/verify_rtl_simulation.sh"
