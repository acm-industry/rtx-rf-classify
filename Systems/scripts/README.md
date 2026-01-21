# Build and Test Scripts

## Purpose

This directory contains automation scripts for building and testing the project across different architectures and environments.

## RISC-V CI Pipeline Integration

The `riscv_build_and_run.sh` script is the core component of the RISC-V CI pipeline. It orchestrates the complete workflow:

1. **Docker image preparation**: Builds the RISC-V toolchain Docker image (cached after first run)
2. **Clean build**: Removes any existing RISC-V build directory to avoid CMake cache mismatches between host paths and container paths
3. **Cross-compilation**: Configures CMake with the RISC-V toolchain and builds the project inside the Docker container
4. **Execution testing**: Runs the compiled RISC-V binary using QEMU user-mode emulation with the appropriate sysroot

The script ensures that:
- Builds are isolated from host system dependencies
- CMake cache paths remain consistent (all configuration happens inside Docker)
- RISC-V binaries are validated through actual execution, not just compilation
- The process is reproducible across different development and CI environments

This script can be used both locally for development and in CI/CD pipelines to validate RISC-V compatibility.

## Files

- `riscv_build_and_run.sh`: Bash script that builds the Docker image, cross-compiles for RISC-V, and executes the binary using QEMU
