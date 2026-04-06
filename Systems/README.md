# Systems

## Purpose

This directory contains the build system, source code, and tooling for cross-compiling and testing C++ code for RISC-V architecture with vector extension (RVV 1.0) support. Two verification pipelines are provided at different levels of abstraction.

## Pipeline 1: Spike ISA Simulation

Validates that the compiler auto-vectorizes operations into RVV instructions and that the vectorized code produces correct results. Uses Docker for isolation and Spike (the RISC-V ISA simulator from the [Ara](https://github.com/pulp-platform/ara) ecosystem) for instruction-accurate simulation.

**Run:** `scripts/spike_build_and_run.sh`

1. Compiles test kernels to assembly and verifies RVV instructions are present
2. Compiles test kernels to a static binary and runs on Spike for correctness
3. Runs Spike with instruction logging to measure the vectorization ratio

See `scripts/README.md` for detailed documentation on each phase.

## Pipeline 2: Ara RTL Simulation

Validates that our kernels execute correctly on the actual [Ara](https://github.com/pulp-platform/ara) hardware model (CVA6 scalar core + Ara vector unit) compiled to a cycle-accurate C++ simulation via Verilator. Provides cycle-accurate performance data and optional FST waveform traces.

**Run:** `scripts/rtl_build_and_run.sh [--trace] [--config <2_lanes|4_lanes|8_lanes|16_lanes>]`

1. Copies bare-metal test kernels into Ara's app directory and compiles with auto-vectorization enabled
2. Runs the binary on the Verilated Ara chip model, reports per-kernel cycle counts and PASS/FAIL

The first Docker build takes 2-4 hours (LLVM toolchain from source, Verilator, hardware verilate). Subsequent runs use the cached image.

See `scripts/README.md` for detailed documentation.

## How the Pipelines Relate

| | Spike (Pipeline 1) | Ara RTL (Pipeline 2) |
|---|---|---|
| **Level** | ISA simulation | Hardware simulation |
| **Speed** | Seconds | Minutes |
| **First build** | ~5 min | ~2-4 hours |
| **Answers** | Are RVV instructions generated and correct? | Do they execute correctly on the Ara chip? How many cycles? |
| **Output** | Instruction counts, vectorization ratio | Cycle-accurate timings, optional waveform traces |

## Directory Structure

- `cmake/`: CMake toolchain configurations for cross-compilation
- `docker/`: Dockerfiles for build environments (Spike and Ara RTL)
- `scripts/`: Automation scripts for building and testing
- `src/`: C++ source code and test kernels
- `build/`: Generated build artifacts (not version controlled)
