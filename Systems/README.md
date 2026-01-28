# Systems

## Purpose

This directory contains the build system, source code, and tooling for cross-compiling and testing C++ code for RISC-V architecture.

## RISC-V CI Pipeline Overview

The RISC-V CI pipeline provides automated validation that the codebase can be successfully compiled for RISC-V 64-bit Linux and executed correctly. The pipeline uses Docker for isolation and QEMU for emulation, ensuring reproducible builds and tests across different host systems.

### Pipeline Components

The CI pipeline consists of several integrated components:

1. **Docker Environment** (`docker/`): Provides an isolated build environment with RISC-V cross-compilation tools and QEMU emulator
2. **CMake Toolchain** (`cmake/toolchains/`): Configures CMake to cross-compile for RISC-V architecture
3. **Build Script** (`scripts/riscv_build_and_run.sh`): Orchestrates the complete build and test workflow
4. **Source Code** (`src/`): C++ code that is compiled and tested for RISC-V compatibility

### Workflow

The pipeline workflow, executed via `scripts/riscv_build_and_run.sh`:

1. Builds a Docker image containing RISC-V toolchain and QEMU
2. Configures CMake with the RISC-V toolchain file
3. Cross-compiles the source code for RISC-V 64-bit Linux
4. Executes the compiled binary using QEMU user-mode emulation
5. Validates that the program runs correctly

This ensures that the codebase maintains RISC-V compatibility and can be deployed to RISC-V-based systems.

## Directory Structure

- `cmake/`: CMake toolchain configurations for cross-compilation
- `docker/`: Dockerfiles for build environments
- `scripts/`: Automation scripts for building and testing
- `src/`: C++ source code
- `build/`: Generated build artifacts (not version controlled)
