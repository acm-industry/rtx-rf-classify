# Docker Build Environment

## Purpose

This directory contains Dockerfiles that define isolated build environments for cross-compilation and testing. Each Dockerfile provides a consistent, reproducible environment for one of the RISC-V verification pipelines.

## Dockerfile.riscv-rvv (Spike ISA Simulation)

Used by `scripts/spike_build_and_run.sh`. Creates a Docker image containing:

1. **Clang 18**: Compiler with mature RISC-V Vector (RVV) auto-vectorization support
2. **RISC-V GCC toolchain**: `gcc-riscv64-linux-gnu` and `g++-riscv64-linux-gnu` for the cross-compilation sysroot, headers, and linker
3. **Spike**: The RISC-V ISA simulator ([riscv-isa-sim](https://github.com/riscv-software-src/riscv-isa-sim)), built from source with full RVV 1.0 support
4. **pk**: The proxy kernel ([riscv-pk](https://github.com/riscv-software-src/riscv-pk)) for handling syscalls when running binaries on Spike
5. **OpenBLAS**: Cross-compiled for RISC-V (static, generic target)

First build: ~5 minutes. Image size: ~2 GB.

## Dockerfile.riscv-rtl (Ara RTL Simulation)

Used by `scripts/rtl_build_and_run.sh`. Creates a Docker image containing the full [Ara](https://github.com/pulp-platform/ara) hardware simulation environment:

1. **Ara repository**: Cloned with LLVM, newlib, and Verilator submodules
2. **RISC-V LLVM toolchain**: Clang + newlib + compiler-rt built from source, targeting `riscv64-unknown-elf` (bare-metal)
3. **Verilator**: RTL simulator built from source (v5.012)
4. **Verilated Ara hardware model**: The full CVA6 + Ara SoC compiled to a cycle-accurate C++ executable via Verilator
5. **Bender**: Hardware IP dependency manager with all IPs checked out and patched

First build: ~2-4 hours (LLVM ~90 min, Verilator ~15 min, hardware verilate ~30 min). Image size: ~15-20 GB. Subsequent runs use the cached image and take only minutes.

## Files

- `Dockerfile.riscv-rvv`: Spike ISA simulation environment (Ubuntu 24.04, Clang, Spike, pk)
- `Dockerfile.riscv-rtl`: Ara RTL simulation environment (Ubuntu 24.04, full Ara toolchain + Verilated hardware)
