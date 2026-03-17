# Systems

## Purpose

This directory contains the build system, source code, and tooling for cross-compiling and testing C++ code for RISC-V architecture with vector extension (RVV 1.0) support.

## RISC-V Vectorization Pipeline Overview

The RISC-V pipeline validates that the compiler auto-vectorizes operations into RVV instructions and that the vectorized code produces correct results. The pipeline uses Docker for isolation and Spike (the RISC-V ISA simulator from the [Ara](https://github.com/pulp-platform/ara) ecosystem) for instruction-accurate simulation.

### Pipeline Components

1. **Docker Environment** (`docker/`): Provides an isolated build environment with Clang, RISC-V cross-compilation tools, Spike, and the proxy kernel (pk)
2. **CMake Toolchain** (`cmake/toolchains/`): Configures CMake to cross-compile for RISC-V with the V (vector) extension enabled
3. **Build Script** (`scripts/spike_build_and_run.sh`): Orchestrates the complete build and verification workflow
4. **Verification Script** (`scripts/verify_rvv_vectorization.sh`): Three-phase pipeline for assembly analysis, runtime correctness, and instruction trace
5. **Source Code** (`src/`): C++ code that is compiled and tested for RISC-V compatibility

### Workflow

The pipeline workflow, executed via `scripts/spike_build_and_run.sh`:

1. Builds a Docker image containing Clang, Spike, and the RISC-V toolchain
2. Compiles test kernels to assembly and verifies RVV instructions are present
3. Compiles test kernels to a static binary and runs on Spike for correctness
4. Runs Spike with instruction logging to measure the vectorization ratio

See `scripts/README.md` for detailed documentation on each phase.

## Directory Structure

- `cmake/`: CMake toolchain configurations for cross-compilation
- `docker/`: Dockerfiles for build environments
- `scripts/`: Automation scripts for building and testing
- `src/`: C++ source code
- `build/`: Generated build artifacts (not version controlled)
