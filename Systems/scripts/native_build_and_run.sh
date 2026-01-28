#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="build/native"

# Clean build directory to avoid cache issues
rm -rf "${ROOT_DIR}/${BUILD_DIR}"

# Configure + build for native architecture
cmake -S "${ROOT_DIR}" -B "${ROOT_DIR}/${BUILD_DIR}" \
  -G "Ninja" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "${ROOT_DIR}/${BUILD_DIR}" -j

# Run the native binary
"${ROOT_DIR}/${BUILD_DIR}/hello"
