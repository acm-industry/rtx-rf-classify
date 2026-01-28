# CMake Toolchain Configuration

## Purpose

This directory contains CMake toolchain files that configure the build system to cross-compile for RISC-V architectures. The toolchain configuration is essential for the CI pipeline that validates RISC-V compatibility.

## RISC-V CI Pipeline Integration

The RISC-V CI pipeline uses the toolchain file `toolchains/riscv64-linux-gnu.cmake` to configure CMake for cross-compilation. This ensures that:

1. **Cross-compilation**: The build system targets RISC-V 64-bit Linux (`riscv64`) instead of the host architecture
2. **Architecture specification**: Sets the RISC-V architecture (`rv64gc`) and ABI (`lp64d`) flags
3. **Compiler selection**: Directs CMake to use the RISC-V cross-compiler toolchain (`riscv64-linux-gnu-gcc`/`riscv64-linux-gnu-g++`)

The CI pipeline (via `scripts/riscv_build_and_run.sh`) passes this toolchain file to CMake using the `-DCMAKE_TOOLCHAIN_FILE` flag, enabling reproducible RISC-V builds in Docker containers.

## Files

- `toolchains/riscv64-linux-gnu.cmake`: CMake toolchain file that configures the RISC-V 64-bit Linux cross-compilation environment
