#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IMAGE_NAME="systems-riscv-toolchain:latest"
BUILD_DIR="build/riscv"

# Build the Docker image (cached after first run)
docker build -f "${ROOT_DIR}/docker/Dockerfile.riscv" -t "${IMAGE_NAME}" "${ROOT_DIR}"

# IMPORTANT: Avoid CMake cache mismatch between host paths and /work in container.
# Ensure this build dir is only ever configured inside Docker.
rm -rf "${ROOT_DIR}/${BUILD_DIR}"

# Configure + build inside container (single-line to avoid quoting issues)
docker run --rm \
  -v "${ROOT_DIR}:/work" \
  -w /work \
  "${IMAGE_NAME}" \
  bash -lc "cmake -S . -B ${BUILD_DIR} -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/riscv64-linux-gnu.cmake && cmake --build ${BUILD_DIR} -j"

# Run inside container (QEMU + sysroot available)
docker run --rm \
  -v "${ROOT_DIR}:/work" \
  -w /work \
  "${IMAGE_NAME}" \
  bash -lc 'SYSROOT=$(riscv64-linux-gnu-g++ -print-sysroot) && qemu-riscv64 -L "$SYSROOT" ./build/riscv/hello'
