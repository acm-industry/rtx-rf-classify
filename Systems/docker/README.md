# Docker Build Environment

## Purpose

This directory contains Dockerfiles that define isolated build environments for cross-compilation and testing. The Docker image provides a consistent, reproducible environment for the RISC-V CI pipeline.

## RISC-V CI Pipeline Integration

The RISC-V CI pipeline uses `Dockerfile.riscv` to create a Docker image containing:

1. **RISC-V toolchain**: `gcc-riscv64-linux-gnu` and `g++-riscv64-linux-gnu` for cross-compilation
2. **Build tools**: CMake and Ninja for build system configuration and execution
3. **QEMU emulator**: `qemu-user` for running RISC-V binaries on non-RISC-V hosts
4. **System dependencies**: Standard build tools and libraries

The CI script (`scripts/riscv_build_and_run.sh`) builds this Docker image and uses it to:
- Configure and build the project for RISC-V architecture
- Execute the compiled binaries using QEMU user-mode emulation
- Ensure consistent results across different host systems

This Docker-based approach eliminates host system dependencies and ensures that RISC-V builds and tests work identically in CI environments and on developer machines.

## Files

- `Dockerfile.riscv`: Dockerfile that sets up an Ubuntu-based environment with RISC-V cross-compilation tools and QEMU
