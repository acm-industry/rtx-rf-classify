# Docker Build Environment

## Purpose

This directory contains Dockerfiles that define isolated build environments for cross-compilation and testing. The Docker image provides a consistent, reproducible environment for the RISC-V vectorization pipeline.

## RISC-V Vectorization Pipeline

The pipeline uses `Dockerfile.riscv-rvv` to create a Docker image containing:

1. **Clang 18**: Compiler with mature RISC-V Vector (RVV) auto-vectorization support
2. **RISC-V GCC toolchain**: `gcc-riscv64-linux-gnu` and `g++-riscv64-linux-gnu` for the cross-compilation sysroot, headers, and linker
3. **Spike**: The RISC-V ISA simulator ([riscv-isa-sim](https://github.com/riscv-software-src/riscv-isa-sim)), built from source with full RVV 1.0 support
4. **pk**: The proxy kernel ([riscv-pk](https://github.com/riscv-software-src/riscv-pk)) for handling syscalls when running binaries on Spike
5. **OpenBLAS**: Cross-compiled for RISC-V (static, generic target)
6. **Build tools**: CMake and Ninja for build system configuration and execution

The pipeline script (`scripts/spike_build_and_run.sh`) builds this Docker image and uses it to compile test kernels, verify RVV instruction emission, and run them on Spike for correctness.

## Files

- `Dockerfile.riscv-rvv`: Dockerfile that sets up an Ubuntu 24.04 environment with Clang, RISC-V cross-compilation tools, Spike, and pk
