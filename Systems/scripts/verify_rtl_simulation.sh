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
#
# Diagnostics added for hang debugging (was: simv could deadlock with no signal):
#   - All log lines are timestamped via ts_log / ts_pipe.
#   - A heartbeat goroutine prints "still running, T=NmSs" every HEARTBEAT_SECS
#     while simv is running, so a stuck terminal is obviously distinguishable
#     from a terminal that's just waiting on a slow Verilator step.
#   - The simv phase runs under `timeout --foreground ${SIMV_TIMEOUT}` so a
#     genuine deadlock (test forgot to write tohost, infinite vsetvli, etc.)
#     terminates cleanly instead of running forever.
#   - On timeout or non-zero exit, we dump the last 200 lines of
#     rtl_sim_output.txt and a `ps -ef` snapshot from inside the container so
#     post-mortem diagnosis doesn't require re-running.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARA_DIR="${ARA_DIR:-/opt/ara}"
ARA_CONFIG="${ARA_CONFIG:-default}"
TRACE="${TRACE:-}"
APP="${APP:-kernels}"

# Phase-2 watchdog.  0 disables.  Set generously: even a small kernel can
# take 5+ minutes of wall time on Verilator.  Override per invocation via
# rtl_build_and_run.sh --simv-timeout.
SIMV_TIMEOUT="${SIMV_TIMEOUT:-1800}"
HEARTBEAT_SECS="${HEARTBEAT_SECS:-30}"

BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
GREY='\033[0;37m'
RESET='\033[0m'

overall_pass=0
overall_fail=0

# ────────────────────────────────────────────────────────────
# Logging helpers
# ────────────────────────────────────────────────────────────

ts_log() {
    printf "${GREY}[%s]${RESET} %b\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

# Pipe stdin through a per-line timestamp prefix.  Keeps `tee` happy by
# reading line-by-line (`IFS= read -r`) so block-buffered producers still get
# their output to the host log even mid-line.  Wraps `make` invocations.
ts_pipe() {
    while IFS= read -r line; do
        printf "${GREY}[%s]${RESET} %s\n" "$(date '+%H:%M:%S')" "$line"
    done
}

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
        printf "${RED}Unknown APP=%s; expected 'kernels' or 'mvp'${RESET}\n" "${APP}" >&2
        exit 1
        ;;
esac

# ────────────────────────────────────────────────────────────
# Toolchain banner
# ────────────────────────────────────────────────────────────
ts_log "${BOLD}Toolchain banner${RESET}"
ts_log "  ARA_DIR        = ${ARA_DIR}"
ts_log "  ARA_CONFIG     = ${ARA_CONFIG}"
ts_log "  APP            = ${APP} (${APP_NAME})"
ts_log "  TRACE          = ${TRACE:-off}"
ts_log "  SIMV_TIMEOUT   = ${SIMV_TIMEOUT}s"
ts_log "  HEARTBEAT_SECS = ${HEARTBEAT_SECS}s"
if command -v verilator >/dev/null 2>&1; then
    ts_log "  verilator      = $(verilator --version 2>&1 | head -n1)"
fi
if [ -x "${ARA_DIR}/install/riscv-llvm/bin/clang" ]; then
    ts_log "  riscv clang    = $("${ARA_DIR}/install/riscv-llvm/bin/clang" --version 2>&1 | head -n1)"
fi
ts_log "  date           = $(date)"
ts_log ""

# ────────────────────────────────────────────────────────────
# Phase 1: Stage sources + compile
# ────────────────────────────────────────────────────────────
printf "\n${BOLD}══════════════════════════════════════════════${RESET}\n"
printf "${BOLD}  Phase 1: App Compilation (%s)${RESET}\n" "${APP}"
printf "${BOLD}══════════════════════════════════════════════${RESET}\n\n"

if [ "${APP}" = "kernels" ]; then
    KERNEL_SRC="${ROOT_DIR}/src/tests/test_rvv_kernels_baremetal.c"
    mkdir -p "${APP_DIR}"
    cp "${KERNEL_SRC}" "${APP_DIR}/main.c"

    ts_log "  source:  ${KERNEL_SRC}"
    ts_log "  target:  ${APP_DIR}/main.c"
    ts_log "  config:  ${ARA_CONFIG}"
    ts_log ""
    ts_log "  Compiling with auto-vectorization enabled (LLVM_V_FLAGS=\"\")..."
else
    # MVP: stage Systems/apps/classify_mvp/ + needed Systems/src/* into APP_DIR
    SYSTEMS_SRC="${ROOT_DIR}/src"
    SYSTEMS_BIN="${ROOT_DIR}/scripts/binaries"
    APP_STAGING="${ROOT_DIR}/apps/classify_mvp"

    if [ ! -f "${SYSTEMS_SRC}/binaries/baremetal_input.bin" ]; then
        ts_log "  ${YELLOW}note${RESET}: no baremetal_input.bin found; running"
        ts_log "  scripts/binaries/generate_baremetal_input.py to bake the fixture..."
        python3 "${ROOT_DIR}/scripts/binaries/generate_baremetal_input.py" \
            --out-dir "${SYSTEMS_SRC}/binaries"
        echo ""
    fi

    rm -rf "${APP_DIR}"
    mkdir -p "${APP_DIR}"

    cp -v "${APP_STAGING}/"main.cpp "${APP_STAGING}/"Makefile "${APP_STAGING}/"README.md \
        "${APP_DIR}/" 2>&1 | sed 's|.*/||;s|^|    |'

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

    if [ -d "${SYSTEMS_SRC}/ExprSystem" ]; then
        mkdir -p "${APP_DIR}/ExprSystem"
        cp -v "${SYSTEMS_SRC}/ExprSystem/"*.h "${APP_DIR}/ExprSystem/" \
            2>&1 | sed 's|.*/||;s|^|    |' || true
    fi

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
    ts_log "  staged ${APP_DIR}"
    ts_log "  config: ${ARA_CONFIG}"
    ts_log ""
    ts_log "  Compiling classify_mvp with auto-vectorization (LLVM_V_FLAGS=\"\")..."
fi

# stdbuf -oL forces line-buffering through tee/ts_pipe even when make is not
# attached to a TTY, which is the difference between "live progress" and "log
# appears to be frozen for 10 minutes".
if LLVM_V_FLAGS="" stdbuf -oL -eL \
        make -C "${ARA_DIR}/apps" "bin/${APP_NAME}" config="${ARA_CONFIG}" 2>&1 | ts_pipe; then
    echo ""
    printf "  ${GREEN}${BOLD}PASS${RESET}: binary compiled successfully\n"
    ts_log "  binary: ${ARA_DIR}/apps/bin/${APP_NAME}"
    overall_pass=$((overall_pass + 1))
else
    rc=$?
    echo ""
    printf "  ${RED}${BOLD}FAIL${RESET}: compilation failed (exit %d)\n" "${rc}"
    overall_fail=$((overall_fail + 1))
    printf "\n${RED}${BOLD}OVERALL: FAIL${RESET} (compilation error, cannot proceed)\n"
    exit 1
fi

# ────────────────────────────────────────────────────────────
# Phase 2: RTL Simulation (Verilated Ara)
# ────────────────────────────────────────────────────────────
printf "\n${BOLD}══════════════════════════════════════════════${RESET}\n"
printf "${BOLD}  Phase 2: RTL Simulation (Verilated Ara)${RESET}\n"
printf "${BOLD}══════════════════════════════════════════════${RESET}\n\n"

SIMV_OUTPUT="${ROOT_DIR}/build/rtl_sim_output.txt"
mkdir -p "$(dirname "${SIMV_OUTPUT}")"
# Truncate any prior log so peek_rtl_sim.sh doesn't tail stale content.
: > "${SIMV_OUTPUT}"

TRACE_FLAG=""
if [ -n "${TRACE}" ]; then
    TRACE_FLAG="trace=1"
    ts_log "  FST waveform tracing: ON"
else
    ts_log "  FST waveform tracing: off (use --trace to enable)"
fi
ts_log "  Running on Verilated Ara chip model..."
ts_log "  log:  ${SIMV_OUTPUT}"
ts_log "  ${YELLOW}If this hangs:${RESET} from another terminal,"
ts_log "    bash ${ROOT_DIR}/scripts/peek_rtl_sim.sh"
ts_log "  to tail the log + inspect simv inside the container."
echo ""

# ── Heartbeat sidecar ─────────────────────────────────────────────────────
# Prints "[T=XmYs] simv still running" every HEARTBEAT_SECS while Phase 2 is
# in flight, plus the most recent log line so it's clear what step Verilator
# is currently chewing on.  Killed in the trap below regardless of exit path.
START_TS=$(date +%s)
heartbeat() {
    while true; do
        sleep "${HEARTBEAT_SECS}"
        local now elapsed mins secs last_line
        now=$(date +%s)
        elapsed=$(( now - START_TS ))
        mins=$(( elapsed / 60 ))
        secs=$(( elapsed % 60 ))
        if [ -s "${SIMV_OUTPUT}" ]; then
            last_line=$(tail -n 1 "${SIMV_OUTPUT}" | tr -d '\r' | cut -c1-160)
        else
            last_line="(no output yet on stdout from simv)"
        fi
        printf "${CYAN}[heartbeat T=%dm%02ds]${RESET} simv running; last: %s\n" \
               "${mins}" "${secs}" "${last_line}"
    done
}
heartbeat &
HEARTBEAT_PID=$!

cleanup() {
    if [ -n "${HEARTBEAT_PID:-}" ] && kill -0 "${HEARTBEAT_PID}" 2>/dev/null; then
        kill "${HEARTBEAT_PID}" 2>/dev/null || true
        wait "${HEARTBEAT_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ── simv invocation, watchdog-protected ───────────────────────────────────
SIMV_CMD=( make -C "${ARA_DIR}/hardware" simv "app=${APP_NAME}" "config=${ARA_CONFIG}" )
if [ -n "${TRACE_FLAG}" ]; then
    SIMV_CMD+=( "${TRACE_FLAG}" )
fi

simv_rc=0
if [ "${SIMV_TIMEOUT}" = "0" ]; then
    ts_log "Watchdog disabled (SIMV_TIMEOUT=0)."
    set +e
    stdbuf -oL -eL "${SIMV_CMD[@]}" 2>&1 | tee "${SIMV_OUTPUT}" | ts_pipe
    simv_rc=${PIPESTATUS[0]}
    set -e
else
    ts_log "Watchdog: simv will be killed after ${SIMV_TIMEOUT}s of wall time."
    set +e
    # --foreground: timeout sends signals to the whole pgrp so verilator dies
    # cleanly along with make.  --kill-after=10: SIGKILL if SIGTERM ignored.
    timeout --foreground --kill-after=10 "${SIMV_TIMEOUT}" \
        stdbuf -oL -eL "${SIMV_CMD[@]}" 2>&1 \
        | tee "${SIMV_OUTPUT}" \
        | ts_pipe
    simv_rc=${PIPESTATUS[0]}
    set -e
fi

cleanup
trap - EXIT

END_TS=$(date +%s)
SIMV_ELAPSED=$(( END_TS - START_TS ))
ts_log "simv phase finished in ${SIMV_ELAPSED}s with exit code ${simv_rc}."

if [ "${simv_rc}" = "124" ] || [ "${simv_rc}" = "137" ]; then
    # 124 = SIGTERM from timeout, 137 = 128+SIGKILL.
    printf "  ${RED}${BOLD}FAIL${RESET}: simv exceeded ${SIMV_TIMEOUT}s watchdog and was killed.\n"
    printf "  ${YELLOW}Last 200 lines of ${SIMV_OUTPUT}:${RESET}\n"
    tail -n 200 "${SIMV_OUTPUT}" | sed 's/^/      /'
    overall_fail=$((overall_fail + 1))
elif [ "${simv_rc}" -ne 0 ]; then
    printf "  ${RED}${BOLD}FAIL${RESET}: simv returned non-zero exit code ${simv_rc}.\n"
    printf "  ${YELLOW}Last 200 lines of ${SIMV_OUTPUT}:${RESET}\n"
    tail -n 200 "${SIMV_OUTPUT}" | sed 's/^/      /'
    overall_fail=$((overall_fail + 1))
else
    # ── Per-app result extraction ─────────────────────────────────────────
    if [ "${APP}" = "kernels" ]; then
        if grep -q "^0 failure" "${SIMV_OUTPUT}"; then
            printf "  ${GREEN}${BOLD}PASS${RESET}: all runtime checks passed on Verilated Ara\n"
            overall_pass=$((overall_pass + 1))
        else
            FAIL_COUNT=$(grep -oP '^\d+(?= failure)' "${SIMV_OUTPUT}" 2>/dev/null || echo "?")
            printf "  ${RED}${BOLD}FAIL${RESET}: %s runtime check(s) failed on Verilated Ara\n" "${FAIL_COUNT}"
            overall_fail=$((overall_fail + 1))
        fi
    else
        PRED_LINE=$(grep -oE '\[ARGMAX-START\][0-9a-f]*\[ARGMAX-END\]' "${SIMV_OUTPUT}" | head -n 1 || true)
        if [ -z "${PRED_LINE}" ]; then
            printf "  ${RED}${BOLD}FAIL${RESET}: could not find [ARGMAX-START]...[ARGMAX-END] in simv output\n"
            overall_fail=$((overall_fail + 1))
        else
            HEX_PAYLOAD="${PRED_LINE#\[ARGMAX-START\]}"
            HEX_PAYLOAD="${HEX_PAYLOAD%\[ARGMAX-END\]}"
            PRED_BIN="${ROOT_DIR}/build/classify_mvp_predicted.bin"
            printf '%s' "${HEX_PAYLOAD}" | xxd -r -p > "${PRED_BIN}"
            EXPECTED_BIN="${APP_DIR}/baremetal_expected.bin"

            if [ -f "${EXPECTED_BIN}" ]; then
                if cmp -s "${PRED_BIN}" "${EXPECTED_BIN}"; then
                    printf "  ${GREEN}${BOLD}PASS${RESET}: predictions match baremetal_expected.bin (%d bytes)\n" \
                        "$(wc -c < "${PRED_BIN}")"
                    overall_pass=$((overall_pass + 1))
                else
                    printf "  ${RED}${BOLD}FAIL${RESET}: predictions differ from baremetal_expected.bin\n"
                    echo "    expected (xxd):"; xxd "${EXPECTED_BIN}" | sed 's/^/      /'
                    echo "    actual   (xxd):"; xxd "${PRED_BIN}"     | sed 's/^/      /'
                    overall_fail=$((overall_fail + 1))
                fi
            else
                printf "  ${YELLOW}${BOLD}SKIP${RESET}: no baremetal_expected.bin staged; got predictions:\n"
                xxd "${PRED_BIN}" | sed 's/^/      /'
                overall_pass=$((overall_pass + 1))
            fi

            CYCLES=$(grep -oE '\[CYCLES\][^[:space:]]*[[:space:]]+[0-9]+' "${SIMV_OUTPUT}" | tail -n 1 || true)
            if [ -n "${CYCLES}" ]; then
                printf "  ${CYAN}%s${RESET}\n" "${CYCLES}"
            fi
        fi
    fi
fi

# Report FST trace location
if [ -n "${TRACE}" ]; then
    FST_FILE="${ARA_DIR}/hardware/build/verilator/ara_tb_verilator.fst"
    if [ -f "${FST_FILE}" ]; then
        echo ""
        printf "  ${CYAN}FST waveform trace:${RESET} %s\n" "${FST_FILE}"
        echo "  Open with: gtkwave ${FST_FILE}"
    fi
fi

# ────────────────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────────────────
printf "\n${BOLD}══════════════════════════════════════════════${RESET}\n"
printf "${BOLD}  Summary${RESET}\n"
printf "${BOLD}══════════════════════════════════════════════${RESET}\n\n"
printf "  app:           %s (%s)\n" "${APP}" "${APP_NAME}"
printf "  simv elapsed:  %ds\n" "${SIMV_ELAPSED}"
printf "  phases passed: ${GREEN}%d${RESET}\n" "${overall_pass}"
printf "  phases failed: ${RED}%d${RESET}\n" "${overall_fail}"
echo ""

if [ "$overall_fail" -gt 0 ]; then
    printf "  ${RED}${BOLD}OVERALL: FAIL${RESET}\n"
    exit 1
else
    printf "  ${GREEN}${BOLD}OVERALL: PASS${RESET}\n"
    exit 0
fi
