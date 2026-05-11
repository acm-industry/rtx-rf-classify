#!/usr/bin/env bash
#
# peek_rtl_sim.sh — observe a running RTL simulation without disturbing it.
#
# Usage:
#   bash Systems/scripts/peek_rtl_sim.sh                 # auto-detect container
#   bash Systems/scripts/peek_rtl_sim.sh ara-rtl-kernels # explicit name
#   bash Systems/scripts/peek_rtl_sim.sh --once          # single-shot diagnostic
#                                                       # then exit (CI-friendly)
#   bash Systems/scripts/peek_rtl_sim.sh --tail          # just `tail -f` the
#                                                       # rtl_sim_output.txt log
#
# Designed to answer the question "is my multi-hour run actually doing
# anything?" without having to interrupt the run to find out.  Reports:
#
#   1. Wall-clock age of the rtl_sim_output.txt file (mtime delta).
#   2. Last 20 lines of stdout from simv.
#   3. Top processes inside the container (verilator/simv/make/clang) with
#      their %CPU, %MEM, and elapsed time so you can tell:
#        - simv eating 100% CPU on one core  → genuinely simulating, just slow
#        - simv at 0% CPU                    → deadlocked (HTIF tohost never set)
#        - make compiling clang.c.cpp        → still in Verilator hardware build
#   4. Whether the [ARGMAX-START]…[ARGMAX-END] sentinel has appeared (MVP) or
#      whether the kernels printed "N failure(s)" (kernels).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIMV_OUTPUT="${ROOT_DIR}/build/rtl_sim_output.txt"

ONESHOT=""
TAIL_ONLY=""
TARGET=""

for arg in "$@"; do
    case "${arg}" in
        --once)  ONESHOT="1" ;;
        --tail)  TAIL_ONLY="1" ;;
        -h|--help)
            sed -n '3,25p' "$0"
            exit 0
            ;;
        *)
            if [ -z "${TARGET}" ]; then TARGET="${arg}"; else
                echo "Unexpected argument: ${arg}" >&2; exit 1
            fi
            ;;
    esac
done

if [ -n "${TAIL_ONLY}" ]; then
    if [ ! -f "${SIMV_OUTPUT}" ]; then
        echo "No log at ${SIMV_OUTPUT} yet; either the run hasn't reached Phase 2 or it was wiped." >&2
        exit 1
    fi
    exec tail -F "${SIMV_OUTPUT}"
fi

# Auto-detect a likely container name if the user didn't pass one.
auto_detect_container() {
    docker ps --format '{{.Names}}' | grep -E '^ara-rtl(-|$)' | head -n1 || true
}

if [ -z "${TARGET}" ]; then
    TARGET="$(auto_detect_container)"
fi

BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
GREY='\033[0;37m'
RESET='\033[0m'

print_section() {
    printf "\n${BOLD}── %s ──${RESET}\n" "$1"
}

snapshot() {
    printf "${GREY}[%s]${RESET} ${BOLD}peek_rtl_sim — diagnostic snapshot${RESET}\n" \
        "$(date '+%Y-%m-%d %H:%M:%S')"

    print_section "Container"
    if [ -z "${TARGET}" ]; then
        printf "  ${YELLOW}No 'ara-rtl-*' container is currently running.${RESET}\n"
        printf "  Available containers (any state):\n"
        docker ps -a --format '    {{.ID}}  {{.Names}}  {{.Status}}' | head -n 20
    else
        printf "  name : %s\n" "${TARGET}"
        if docker ps --format '{{.Names}}' | grep -qx "${TARGET}"; then
            printf "  state: ${GREEN}running${RESET}\n"
            docker inspect --format \
                '  uptime: {{.State.StartedAt}} (status: {{.State.Status}})' \
                "${TARGET}" 2>/dev/null || true
        else
            printf "  state: ${RED}not running${RESET}\n"
        fi
    fi

    print_section "Host log (${SIMV_OUTPUT})"
    if [ -f "${SIMV_OUTPUT}" ]; then
        local size mtime age last_line
        size=$(wc -c < "${SIMV_OUTPUT}" | tr -d ' ')
        mtime=$(date -r "${SIMV_OUTPUT}" '+%Y-%m-%d %H:%M:%S' 2>/dev/null \
                || stat -c '%y' "${SIMV_OUTPUT}" 2>/dev/null \
                || echo unknown)
        age=$(( $(date +%s) - $(stat -f %m "${SIMV_OUTPUT}" 2>/dev/null \
                                   || stat -c %Y "${SIMV_OUTPUT}" 2>/dev/null \
                                   || echo 0) ))
        printf "  size       : %s bytes\n" "${size}"
        printf "  last write : %s  (%ds ago)\n" "${mtime}" "${age}"
        if [ "${age}" -gt 120 ] && [ "${size}" -gt 0 ]; then
            printf "  ${YELLOW}WARN${RESET}: log hasn't been written to in over 2 minutes\n"
            printf "         simv may be stuck.  Check the container processes below.\n"
        fi
        printf "  last 20 lines:\n"
        tail -n 20 "${SIMV_OUTPUT}" | sed 's/^/      /' || true
    else
        printf "  ${YELLOW}log does not exist yet${RESET}\n"
    fi

    print_section "Sentinel detection"
    if [ -f "${SIMV_OUTPUT}" ]; then
        if grep -q '\[ARGMAX-END\]' "${SIMV_OUTPUT}" 2>/dev/null; then
            printf "  ${GREEN}MVP closed sentinel found${RESET} — predictions complete:\n"
            grep -oE '\[ARGMAX-START\][0-9a-f]*\[ARGMAX-END\]' "${SIMV_OUTPUT}" \
                | head -n1 | sed 's/^/      /'
        elif grep -q '\[ARGMAX-START\]' "${SIMV_OUTPUT}" 2>/dev/null; then
            printf "  ${YELLOW}MVP open sentinel found, no close${RESET} — simv may be in serve_inference\n"
        elif grep -qE '^[0-9]+ failure' "${SIMV_OUTPUT}" 2>/dev/null; then
            printf "  ${GREEN}kernels test reported${RESET}: $(grep -E '^[0-9]+ failure' "${SIMV_OUTPUT}" | tail -n1)\n"
        else
            printf "  ${GREY}no MVP sentinels or kernels failure-count yet${RESET}\n"
        fi
    fi

    if [ -n "${TARGET}" ] && docker ps --format '{{.Names}}' | grep -qx "${TARGET}"; then
        print_section "Top simulation processes inside ${TARGET}"
        # Long PIDs + busy CPU shells: pick interesting candidates only.
        docker exec "${TARGET}" sh -c '
            ps -eo pid,pcpu,pmem,etime,stat,comm,args --sort=-pcpu \
                | awk "NR==1 || /verilator|simv|ara_tb|make|clang|cc1plus|ld/ { print }" \
                | head -n 25
        ' 2>/dev/null | sed 's/^/  /' \
            || printf "  ${YELLOW}docker exec failed; container may have just exited${RESET}\n"

        print_section "Disk: where simv writes (Verilator obj dir)"
        docker exec "${TARGET}" sh -c '
            for d in /opt/ara/hardware/build/verilator /opt/ara/hardware/build; do
                if [ -d "$d" ]; then
                    echo "  $d:"
                    ls -lt --time-style=long-iso "$d" 2>/dev/null | head -n 8 | sed "s/^/    /"
                    break
                fi
            done
        ' 2>/dev/null \
            || printf "  ${YELLOW}docker exec failed${RESET}\n"
    fi

    print_section "What to do"
    cat <<EOF
  - simv at ~100% CPU and recent log writes ⇒ ${GREEN}healthy, just slow${RESET}
    Verilator simulates Ara at low kHz; a 4-lane config easily takes 10-30
    minutes per kernel.  Wait, or rerun with --simv-timeout for a hard cap.

  - simv at 0% CPU and no log writes for >2 min ⇒ ${RED}deadlocked${RESET}
    Most likely: bare-metal program never wrote tohost (forgot exit, infinite
    loop, or vsetvli VL=0 spin).  Recover with:
        docker kill ${TARGET:-<container>}
    then bisect by reducing the test workload or enabling --trace and
    inspecting the FST in gtkwave.

  - make still compiling .cpp.o files ⇒ ${YELLOW}Verilator C++ build phase${RESET}
    The first simv build per (config, app) compiles thousands of TUs and can
    take 30+ minutes.  This is one-shot; subsequent runs are fast.

  - Live-tail the simv log:
        bash $0 --tail
EOF
}

if [ -n "${ONESHOT}" ]; then
    snapshot
    exit 0
fi

# Default: snapshot every 30s, Ctrl-C to stop.
while true; do
    clear 2>/dev/null || printf '\n\n'
    snapshot
    printf "\n${GREY}refreshing in 30s; Ctrl-C to exit, --once for single-shot${RESET}\n"
    sleep 30
done
