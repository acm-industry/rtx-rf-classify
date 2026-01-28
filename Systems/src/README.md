# Source Code

## Purpose

This directory contains the C++ source code for the Systems project. The code is designed to be compilable for both native architectures and RISC-V via cross-compilation.

## RISC-V CI Pipeline Integration

The source code in this directory is compiled for RISC-V architecture as part of the CI pipeline validation process. The pipeline:

1. **Cross-compiles** the source code using the RISC-V toolchain configured in `cmake/toolchains/riscv64-linux-gnu.cmake`
2. **Links statically** (on non-MSVC platforms) to ensure the binary is self-contained
3. **Tests execution** by running the compiled RISC-V binary using QEMU emulation

This ensures that the codebase is compatible with RISC-V architecture and can be successfully built and executed in RISC-V environments. The CI pipeline validates that:
- The code compiles without errors for RISC-V
- The binary executes correctly when emulated
- No architecture-specific assumptions are made in the code

## Files

- `main.cpp`: Main entry point for the application
