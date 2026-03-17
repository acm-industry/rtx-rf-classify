# CMake Toolchain Configuration

## Purpose

This directory contains CMake toolchain files that configure the build system to cross-compile for RISC-V architectures with vector extension support.

## RISC-V Vectorization Pipeline

The pipeline uses the toolchain file `toolchains/riscv64-rvv.cmake` to configure CMake for cross-compilation. This ensures that:

1. **Cross-compilation**: The build system targets RISC-V 64-bit Linux (`riscv64`) instead of the host architecture
2. **Vector extension**: Sets the RISC-V architecture to `rv64gcv` (includes the V vector extension) and ABI to `lp64d`
3. **Compiler selection**: Directs CMake to use the RISC-V cross-compiler toolchain (`riscv64-linux-gnu-gcc`/`riscv64-linux-gnu-g++`)

The pipeline (via `scripts/spike_build_and_run.sh`) uses this toolchain file when cross-compiling the full project with vector instructions enabled.

## Files

- `toolchains/riscv64-rvv.cmake`: CMake toolchain file that configures RISC-V 64-bit cross-compilation with RVV 1.0 vector extension
