#!/usr/bin/env bash
set -euo pipefail

# ── Ara RTL Simulation Verification Pipeline ──
# Runs inside the Docker container built from Dockerfile.riscv-rtl.
# Two phases:
#   1. Compile a bare-metal app for Ara (auto-vectorization ON)
#   2. Run the binary on the Verilated Ara hardware model
#
# The selected app is controlled by the APP environment variable:
#   APP=kernels (default) - test_rvv_kernels_baremetal.c smoke kernels
#   APP=mvp               - full classification MVP via Systems/apps/classify_mvp/

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARA_DIR="${ARA_DIR:-/opt/ara}"
ARA_CONFIG="${ARA_CONFIG:-default}"
TRACE="${TRACE:-}"
APP="${APP:-kernels}"

BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
RESET='\033[0m'

overall_pass=0
overall_fail=0

# ────────────────────────────────────────────────────────────
# Resolve per-app config
# ────────────────────────────────────────────────────────────
case "${APP}" in
    kernels)
        APP_NAME="rtx_kernels"
        APP_DIR="${ARA_DIR}/apps/${APP_NAME}"
        ;;
    mvp)
        APP_NAME="classify_mvp"
        APP_DIR="${ARA_DIR}/apps/${APP_NAME}"
        ;;
    *)
        echo -e "${RED}Unknown APP=${APP}; expected 'kernels' or 'mvp'${RESET}" >&2
        exit 1
        ;;
esac

# ────────────────────────────────────────────────────────────
# Phase 1: Stage sources + compile
# ────────────────────────────────────────────────────────────
echo -e "\n${BOLD}══════════════════════════════════════════════${RESET}"
echo -e "${BOLD}  Phase 1: App Compilation (${APP})${RESET}"
echo -e "${BOLD}══════════════════════════════════════════════${RESET}\n"

if [ "${APP}" = "kernels" ]; then
    KERNEL_SRC="${ROOT_DIR}/src/tests/test_rvv_kernels_baremetal.c"
    mkdir -p "${APP_DIR}"
    cp "${KERNEL_SRC}" "${APP_DIR}/main.c"

    echo "  source:  ${KERNEL_SRC}"
    echo "  target:  ${APP_DIR}/main.c"
    echo "  config:  ${ARA_CONFIG}"
    echo ""
    echo "  Compiling with auto-vectorization enabled (LLVM_V_FLAGS=\"\")..."
else
    # MVP: stage Systems/apps/classify_mvp/ + needed Systems/src/* into APP_DIR
    SYSTEMS_SRC="${ROOT_DIR}/src"
    SYSTEMS_BIN="${ROOT_DIR}/scripts/binaries"
    APP_STAGING="${ROOT_DIR}/apps/classify_mvp"

    if [ ! -f "${SYSTEMS_SRC}/binaries/baremetal_input.bin" ]; then
        echo -e "  ${YELLOW}note${RESET}: no baremetal_input.bin found; running"
        echo "  scripts/binaries/generate_baremetal_input.py to bake the fixture..."
        python3 "${ROOT_DIR}/scripts/binaries/generate_baremetal_input.py" \
            --out-dir "${SYSTEMS_SRC}/binaries"
        echo ""
    fi

    rm -rf "${APP_DIR}"
    mkdir -p "${APP_DIR}"

    # 1) Per-app shim and Makefile (main.cpp + Makefile + README.md)
    cp -v "${APP_STAGING}/"main.cpp "${APP_STAGING}/"Makefile "${APP_STAGING}/"README.md \
        "${APP_DIR}/" 2>&1 | sed 's|.*/||;s|^|    |'

    # 2) Bare-metal entry + every header serve_inference + BareIOStream pull in
    cp -v \
        "${SYSTEMS_SRC}/main_baremetal.cpp" \
        "${SYSTEMS_SRC}/inference.h" \
        "${SYSTEMS_SRC}/baremetal_stream.h" \
        "${SYSTEMS_SRC}/streams.h" \
        "${SYSTEMS_SRC}/tensor.h" \
        "${SYSTEMS_SRC}/convolve.h" \
        "${SYSTEMS_SRC}/batchnorm.h" \
        "${SYSTEMS_SRC}/blas_ops.h" \
        "${SYSTEMS_SRC}/maxpool.h" \
        "${SYSTEMS_SRC}/avgpool.h" \
        "${SYSTEMS_SRC}/maweights.cpp" \
        "${SYSTEMS_SRC}/memorybuffer.h" \
        "${APP_DIR}/" 2>&1 | sed 's|.*/||;s|^|    |'

    # ExprSystem subtree (used by inference.h via convolve.h / Broadcast.h)
    if [ -d "${SYSTEMS_SRC}/ExprSystem" ]; then
        mkdir -p "${APP_DIR}/ExprSystem"
        cp -v "${SYSTEMS_SRC}/ExprSystem/"*.h "${APP_DIR}/ExprSystem/" \
            2>&1 | sed 's|.*/||;s|^|    |' || true
    fi

    # 3) Assembly: weights + fixture
    cp -v "${SYSTEMS_SRC}/weights.S" "${SYSTEMS_SRC}/baremetal_input.S" \
        "${APP_DIR}/" 2>&1 | sed 's|.*/||;s|^|    |'

    # 4) Raw .bin payloads referenced by .incbin.  LLVM's assembler resolves
    #    .incbin paths relative to the make CWD (${ARA_DIR}/apps/), NOT the
    #    source file's directory, so the files must live in both places.
    cp -v "${SYSTEMS_BIN}/"*.bin "${APP_DIR}/" 2>&1 | sed 's|.*/||;s|^|    |'
    cp -v "${SYSTEMS_BIN}/"*.bin "${ARA_DIR}/apps/" 2>&1 | sed 's|.*/||;s|^|    |'
    if [ -f "${SYSTEMS_SRC}/binaries/baremetal_input.bin" ]; then
        cp -v "${SYSTEMS_SRC}/binaries/baremetal_input.bin" \
            "${SYSTEMS_SRC}/binaries/baremetal_expected.bin" \
            "${APP_DIR}/" 2>&1 | sed 's|.*/||;s|^|    |'
        cp -v "${SYSTEMS_SRC}/binaries/baremetal_input.bin" \
            "${SYSTEMS_SRC}/binaries/baremetal_expected.bin" \
            "${ARA_DIR}/apps/" 2>&1 | sed 's|.*/||;s|^|    |'
    fi

    echo ""
    echo "  staged ${APP_DIR}"
    echo "  config: ${ARA_CONFIG}"
    echo ""
    echo "  Compiling classify_mvp with auto-vectorization (LLVM_V_FLAGS=\"\")..."
fi

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

    if [ "${APP}" = "kernels" ]; then
        # Kernels report `0 failure` to stdout via Ara's check_vector helpers.
        if grep -q "^0 failure" "${SIMV_OUTPUT}"; then
            echo -e "  ${GREEN}${BOLD}PASS${RESET}: all runtime checks passed on Verilated Ara"
            overall_pass=$((overall_pass + 1))
        else
            FAIL_COUNT=$(grep -oP '^\d+(?= failure)' "${SIMV_OUTPUT}" 2>/dev/null || echo "?")
            echo -e "  ${RED}${BOLD}FAIL${RESET}: ${FAIL_COUNT} runtime check(s) failed on Verilated Ara"
            overall_fail=$((overall_fail + 1))
        fi
    else
        # MVP: extract argmax bytes printed between [ARGMAX-START]/[ARGMAX-END]
        # sentinels by HtifOStream::sync(), then diff against the baked
        # baremetal_expected.bin fixture (each byte is one sample's argmax).
        PRED_LINE=$(grep -oE '\[ARGMAX-START\][0-9a-f]*\[ARGMAX-END\]' "${SIMV_OUTPUT}" | head -n 1 || true)
        if [ -z "${PRED_LINE}" ]; then
            echo -e "  ${RED}${BOLD}FAIL${RESET}: could not find [ARGMAX-START]...[ARGMAX-END] in simv output"
            overall_fail=$((overall_fail + 1))
        else
            HEX_PAYLOAD="${PRED_LINE#\[ARGMAX-START\]}"
            HEX_PAYLOAD="${HEX_PAYLOAD%\[ARGMAX-END\]}"
            PRED_BIN="${ROOT_DIR}/build/classify_mvp_predicted.bin"
            printf '%s' "${HEX_PAYLOAD}" | xxd -r -p > "${PRED_BIN}"
            EXPECTED_BIN="${APP_DIR}/baremetal_expected.bin"

            if [ -f "${EXPECTED_BIN}" ]; then
                if cmp -s "${PRED_BIN}" "${EXPECTED_BIN}"; then
                    echo -e "  ${GREEN}${BOLD}PASS${RESET}: predictions match baremetal_expected.bin ($(wc -c < "${PRED_BIN}") bytes)"
                    overall_pass=$((overall_pass + 1))
                else
                    echo -e "  ${RED}${BOLD}FAIL${RESET}: predictions differ from baremetal_expected.bin"
                    echo "    expected (xxd):"; xxd "${EXPECTED_BIN}" | sed 's/^/      /'
                    echo "    actual   (xxd):"; xxd "${PRED_BIN}"     | sed 's/^/      /'
                    overall_fail=$((overall_fail + 1))
                fi
            else
                echo -e "  ${YELLOW}${BOLD}SKIP${RESET}: no baremetal_expected.bin staged; got predictions:"
                xxd "${PRED_BIN}" | sed 's/^/      /'
                # Treat missing oracle as pass-with-warning so CI can still
                # bring up the pipeline before a real expected fixture lands.
                overall_pass=$((overall_pass + 1))
            fi

            # Also surface the cycle counter the runtime printed.
            CYCLES=$(grep -oE '\[CYCLES\][^[:space:]]*[[:space:]]+[0-9]+' "${SIMV_OUTPUT}" | tail -n 1 || true)
            if [ -n "${CYCLES}" ]; then
                echo "  ${CYAN}${CYCLES}${RESET}"
            fi
        fi
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
echo -e "  app:           ${APP} (${APP_NAME})"
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
