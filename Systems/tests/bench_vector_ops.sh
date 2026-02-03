#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${SCRIPT_DIR}/../src/vector/tests/bench_vector_ops.sh"

if [ ! -x "${TARGET}" ]; then
  echo "Target script not found or not executable: ${TARGET}" >&2
  exit 1
fi

exec "${TARGET}" "$@"
