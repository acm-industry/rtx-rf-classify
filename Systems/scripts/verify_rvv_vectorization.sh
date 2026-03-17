#!/usr/bin/env bash
set -euo pipefail

# ── RVV Vectorization Verification Pipeline ──
# Runs inside the Docker container built from Dockerfile.riscv-rvv.
# Three phases:
#   1. Assembly analysis  – compile to .s, count RVV instructions per kernel
#   2. Runtime correctness – compile to binary, run on Spike, check output
#   3. Instruction trace   – run Spike -l on a representative kernel, report vec ratio

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KERNEL_SRC="${ROOT_DIR}/src/tests/test_rvv_kernels.cpp"

# Clang 18 (shipped with Ubuntu 24.04) has mature RVV auto-vectorization.
# GCC 13's RISC-V V backend lacks vector type mappings for float, so it
# cannot auto-vectorize at all.  We keep GCC installed for sysroot/linker.
CXX="clang++"
COMMON_FLAGS=(
    --target=riscv64-linux-gnu --gcc-toolchain=/usr --sysroot=/usr/riscv64-linux-gnu
    -std=c++23 -O3 -march=rv64gcv -mabi=lp64d
    -ffast-math
)

SPIKE="spike"
PK_PATH="/opt/riscv/riscv64-linux-gnu/bin/pk"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
RESET='\033[0m'

overall_pass=0
overall_fail=0

# ────────────────────────────────────────────────────────────
# Phase 1: Assembly Analysis
# ────────────────────────────────────────────────────────────
echo -e "\n${BOLD}══════════════════════════════════════════════${RESET}"
echo -e "${BOLD}  Phase 1: Assembly Analysis (RVV Instruction Check)${RESET}"
echo -e "${BOLD}══════════════════════════════════════════════${RESET}\n"

ASM_FILE="${TMP_DIR}/test_rvv_kernels.s"
echo "compiling ${KERNEL_SRC} to assembly..."
"${CXX}" "${COMMON_FLAGS[@]}" -S "${KERNEL_SRC}" -o "${ASM_FILE}"

# RVV instruction categories to search for
declare -A RVV_CATEGORIES=(
    ["setup"]="vsetvli|vsetivli"
    ["load_store"]="vle[0-9]+\\.v|vse[0-9]+\\.v|vlse[0-9]+\\.v|vsse[0-9]+\\.v"
    ["arith"]="vfadd\\.[vf]+|vfsub\\.[vf]+|vfmul\\.[vf]+|vfdiv\\.[vf]+"
    ["fma"]="vfmadd\\.[vf]+|vfmacc\\.[vf]+|vfnmadd\\.[vf]+|vfnmacc\\.[vf]+|vfmsub\\.[vf]+|vfnmsub\\.[vf]+"
    ["reduce"]="vfredosum|vfredusum|vfredmax|vfredmin"
    ["integer"]="vadd\\.[vi]+|vmul\\.[vi]+|vsll\\.[vi]+|vsrl\\.[vi]+|vand\\.[vi]+"
    ["compare"]="vmflt|vmfle|vmfgt|vmfge|vmfeq|vmfne|vmslt|vmsle"
    ["mask"]="vmand|vmor|vmnot|vmxor"
)

echo ""
printf "  ${BOLD}%-16s  %s${RESET}\n" "Category" "Count"
printf "  %-16s  %s\n" "────────────────" "─────"

total_rvv=0
for category in setup load_store arith fma reduce integer compare mask; do
    pattern="${RVV_CATEGORIES[$category]}"
    count=$(grep -Ec "${pattern}" "${ASM_FILE}" 2>/dev/null || true)
    total_rvv=$((total_rvv + count))
    if [ "$count" -gt 0 ]; then
        printf "  ${GREEN}%-16s  %d${RESET}\n" "$category" "$count"
    else
        printf "  %-16s  %d\n" "$category" "$count"
    fi
done

echo ""
if [ "$total_rvv" -gt 0 ]; then
    echo -e "  ${GREEN}${BOLD}PASS${RESET}: $total_rvv RVV instructions found in assembly"
    overall_pass=$((overall_pass + 1))
else
    echo -e "  ${RED}${BOLD}FAIL${RESET}: no RVV instructions found in assembly"
    overall_fail=$((overall_fail + 1))
fi

# Per-kernel breakdown: extract function labels and count RVV instructions per function
echo ""
echo -e "  ${BOLD}Per-kernel RVV instruction counts:${RESET}"
printf "  %-24s  %s\n" "Kernel" "RVV instrs"
printf "  %-24s  %s\n" "────────────────────────" "──────────"

rvv_regex='vsetvli|vsetivli|vle[0-9]+\.v|vse[0-9]+\.v|vfadd\.|vfsub\.|vfmul\.|vfdiv\.|vfmadd\.|vfmacc\.|vfnmadd\.|vfnmacc\.|vfredosum|vfredusum|vfredmax|vadd\.|vmul\.|vmflt|vmfle|vmfgt'

for func in kernel_add kernel_dot kernel_matvec kernel_fma kernel_conv1d kernel_relu kernel_maxpool kernel_linear; do
    # Clang emits C++-mangled names; GCC may emit plain names.
    # Match any label line containing the kernel name, up to .Lfunc_end (Clang) or .size (GCC).
    func_asm=$(sed -n "/^_Z.*${func}.*:/,/^\\.Lfunc_end/p" "${ASM_FILE}" 2>/dev/null || true)
    if [ -z "$func_asm" ]; then
        func_asm=$(sed -n "/^${func}:/,/^\\.size.*${func}/p" "${ASM_FILE}" 2>/dev/null || true)
    fi
    fcount=$(echo "$func_asm" | grep -Ec "${rvv_regex}" 2>/dev/null || true)
    if [ "$fcount" -gt 0 ]; then
        printf "  ${GREEN}%-24s  %d${RESET}\n" "$func" "$fcount"
    else
        printf "  %-24s  %d\n" "$func" "$fcount"
    fi
done

# ────────────────────────────────────────────────────────────
# Phase 2: Runtime Correctness (Spike Execution)
# ────────────────────────────────────────────────────────────
echo -e "\n${BOLD}══════════════════════════════════════════════${RESET}"
echo -e "${BOLD}  Phase 2: Runtime Correctness (Spike)${RESET}"
echo -e "${BOLD}══════════════════════════════════════════════${RESET}\n"

BIN_FILE="${TMP_DIR}/test_rvv_kernels"
echo "compiling to static binary..."
"${CXX}" "${COMMON_FLAGS[@]}" -static "${KERNEL_SRC}" -o "${BIN_FILE}" -lm

echo "running on Spike..."
SPIKE_OUTPUT="${TMP_DIR}/spike_output.txt"
if "${SPIKE}" --isa=rv64gcv "${PK_PATH}" "${BIN_FILE}" > "${SPIKE_OUTPUT}" 2>/dev/null; then
    echo ""
    cat "${SPIKE_OUTPUT}"
    echo ""

    if grep -q "^0 failure" "${SPIKE_OUTPUT}"; then
        echo -e "  ${GREEN}${BOLD}PASS${RESET}: all runtime checks passed on Spike"
        overall_pass=$((overall_pass + 1))
    else
        echo -e "  ${RED}${BOLD}FAIL${RESET}: some runtime checks failed on Spike"
        overall_fail=$((overall_fail + 1))
    fi
else
    echo -e "  ${RED}${BOLD}FAIL${RESET}: Spike execution returned non-zero exit code"
    cat "${SPIKE_OUTPUT}" 2>/dev/null || true
    overall_fail=$((overall_fail + 1))
fi

# ────────────────────────────────────────────────────────────
# Phase 3: Instruction Trace Analysis
# ────────────────────────────────────────────────────────────
echo -e "\n${BOLD}══════════════════════════════════════════════${RESET}"
echo -e "${BOLD}  Phase 3: Instruction Trace Analysis${RESET}"
echo -e "${BOLD}══════════════════════════════════════════════${RESET}\n"

TRACE_FILE="${TMP_DIR}/spike_trace.log"
echo "running Spike with instruction logging (this may take a moment)..."

# Spike logs instructions to stderr with -l
"${SPIKE}" -l --isa=rv64gcv "${PK_PATH}" "${BIN_FILE}" > /dev/null 2> "${TRACE_FILE}" || true

TOTAL_INSTRS=$(wc -l < "${TRACE_FILE}" | tr -d ' ')

# Count vector instructions in the trace
VEC_TRACE_REGEX='vsetvli|vsetivli|vle[0-9]+\.v|vse[0-9]+\.v|vlse|vsse|vfadd|vfsub|vfmul|vfdiv|vfmadd|vfmacc|vfnmadd|vfnmacc|vfmsub|vfnmsub|vfredosum|vfredusum|vfredmax|vfredmin|vadd|vsub|vmul|vsll|vsrl|vsra|vand|vor|vxor|vmseq|vmsne|vmslt|vmsle|vmflt|vmfle|vmfgt|vmfge'
VEC_INSTRS=$(grep -Ec "${VEC_TRACE_REGEX}" "${TRACE_FILE}" 2>/dev/null || true)
VEC_INSTRS="${VEC_INSTRS:-0}"

if [ "$TOTAL_INSTRS" -gt 0 ]; then
    RATIO=$(awk -v vec="$VEC_INSTRS" -v total="$TOTAL_INSTRS" 'BEGIN { printf "%.1f", (vec / total) * 100 }')
    echo "  total instructions executed:  ${TOTAL_INSTRS}"
    echo "  vector instructions executed: ${VEC_INSTRS}"
    echo -e "  ${CYAN}vectorization ratio:           ${RATIO}%${RESET}"
    overall_pass=$((overall_pass + 1))
else
    echo -e "  ${RED}${BOLD}FAIL${RESET}: no instructions captured in trace"
    overall_fail=$((overall_fail + 1))
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
