#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IMAGE_NAME="systems-riscv-rvv:latest"

# Build the Docker image (Spike + pk + GCC with RVV support)
echo "Building Docker image (first run will take a while for Spike/pk compilation)..."
docker build -f "${ROOT_DIR}/docker/Dockerfile.riscv-rvv" -t "${IMAGE_NAME}" "${ROOT_DIR}"

# Run the RVV vectorization verification pipeline inside the container
docker run --rm \
  -v "${ROOT_DIR}:/work" \
  -w /work \
  "${IMAGE_NAME}" \
  bash -lc "./scripts/verify_rvv_vectorization.sh"
