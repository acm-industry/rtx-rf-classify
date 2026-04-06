#!/usr/bin/env bash
set -euo pipefail

# ── Ara RTL Simulation Verification Pipeline ──
# Runs inside the Docker container built from Dockerfile.riscv-rtl.
# Two phases:
#   1. Compile the bare-metal test kernel as an Ara app (auto-vectorization ON)
#   2. Run the binary on the Verilated Ara hardware model

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARA_DIR="${ARA_DIR:-/opt/ara}"
ARA_CONFIG="${ARA_CONFIG:-default}"
TRACE="${TRACE:-}"

APP_NAME="rtx_kernels"
KERNEL_SRC="${ROOT_DIR}/src/tests/test_rvv_kernels_baremetal.c"

BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
RESET='\033[0m'

overall_pass=0
overall_fail=0

# ────────────────────────────────────────────────────────────
# Phase 1: App Compilation
# ────────────────────────────────────────────────────────────
echo -e "\n${BOLD}══════════════════════════════════════════════${RESET}"
echo -e "${BOLD}  Phase 1: App Compilation${RESET}"
echo -e "${BOLD}══════════════════════════════════════════════${RESET}\n"

APP_DIR="${ARA_DIR}/apps/${APP_NAME}"
mkdir -p "${APP_DIR}"
cp "${KERNEL_SRC}" "${APP_DIR}/main.c"

echo "  source:  ${KERNEL_SRC}"
echo "  target:  ${APP_DIR}/main.c"
echo "  config:  ${ARA_CONFIG}"
echo ""
echo "  Compiling with auto-vectorization enabled (LLVM_V_FLAGS=\"\")..."

# LLVM_V_FLAGS="" removes -fno-vectorize and -scalable-vectorization=off,
# letting -O3 -ffast-math (already in RISCV_FLAGS) drive the auto-vectorizer.
if LLVM_V_FLAGS="" make -C "${ARA_DIR}/apps" "bin/${APP_NAME}" config="${ARA_CONFIG}" 2>&1; then
    echo ""
    echo -e "  ${GREEN}${BOLD}PASS${RESET}: binary compiled successfully"
    echo "  binary: ${ARA_DIR}/apps/bin/${APP_NAME}"
    overall_pass=$((overall_pass + 1))
else
    echo ""
    echo -e "  ${RED}${BOLD}FAIL${RESET}: compilation failed"
    overall_fail=$((overall_fail + 1))
    echo -e "\n${RED}${BOLD}OVERALL: FAIL${RESET} (compilation error, cannot proceed)"
    exit 1
fi

# ────────────────────────────────────────────────────────────
# Phase 2: RTL Simulation (Verilated Ara)
# ────────────────────────────────────────────────────────────
echo -e "\n${BOLD}══════════════════════════════════════════════${RESET}"
echo -e "${BOLD}  Phase 2: RTL Simulation (Verilated Ara)${RESET}"
echo -e "${BOLD}══════════════════════════════════════════════${RESET}\n"

SIMV_OUTPUT="${ROOT_DIR}/build/rtl_sim_output.txt"
mkdir -p "$(dirname "${SIMV_OUTPUT}")"

TRACE_FLAG=""
if [ -n "${TRACE}" ]; then
    TRACE_FLAG="trace=1"
    echo "  FST waveform tracing: ON"
else
    echo "  FST waveform tracing: off (use --trace to enable)"
fi
echo "  Running on Verilated Ara chip model..."
echo ""

if make -C "${ARA_DIR}/hardware" simv \
        "app=${APP_NAME}" \
        "config=${ARA_CONFIG}" \
        ${TRACE_FLAG} \
        2>&1 | tee "${SIMV_OUTPUT}"; then

    echo ""

    if grep -q "^0 failure" "${SIMV_OUTPUT}"; then
        echo -e "  ${GREEN}${BOLD}PASS${RESET}: all runtime checks passed on Verilated Ara"
        overall_pass=$((overall_pass + 1))
    else
        FAIL_COUNT=$(grep -oP '^\d+(?= failure)' "${SIMV_OUTPUT}" 2>/dev/null || echo "?")
        echo -e "  ${RED}${BOLD}FAIL${RESET}: ${FAIL_COUNT} runtime check(s) failed on Verilated Ara"
        overall_fail=$((overall_fail + 1))
    fi
else
    echo ""
    echo -e "  ${RED}${BOLD}FAIL${RESET}: Verilator simulation returned non-zero exit code"
    overall_fail=$((overall_fail + 1))
fi

# Report FST trace location
if [ -n "${TRACE}" ]; then
    FST_FILE="${ARA_DIR}/hardware/build/verilator/ara_tb_verilator.fst"
    if [ -f "${FST_FILE}" ]; then
        echo ""
        echo -e "  ${CYAN}FST waveform trace:${RESET} ${FST_FILE}"
        echo "  Open with: gtkwave ${FST_FILE}"
    fi
fi

# ────────────────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────────────────
echo -e "\n${BOLD}══════════════════════════════════════════════${RESET}"
echo -e "${BOLD}  Summary${RESET}"
echo -e "${BOLD}══════════════════════════════════════════════${RESET}\n"
echo -e "  phases passed: ${GREEN}${overall_pass}${RESET}"
echo -e "  phases failed: ${RED}${overall_fail}${RESET}"
echo ""

if [ "$overall_fail" -gt 0 ]; then
    echo -e "  ${RED}${BOLD}OVERALL: FAIL${RESET}"
    exit 1
else
    echo -e "  ${GREEN}${BOLD}OVERALL: PASS${RESET}"
    exit 0
fi
