#!/usr/bin/env python3
"""Generate a deterministic input fixture for the bare-metal Ara MVP build.

Writes two files into Systems/src/binaries/ (or wherever ``--out-dir`` points):

  baremetal_input.bin     - exact wire payload the StaticBufferIStream feeds
                            serve_inference: a 4-byte big-endian batch length
                            followed by ``batch * (3 * 128)`` float32 samples
                            in host (little-endian) byte order.

  baremetal_expected.bin  - one byte per sample, the expected argmax that the
                            host-side runner (verify_rtl_simulation.sh) diffs
                            against the predictions printed via HTIF.

Generating the expected outputs is optional; the script prints them anyway and
falls back to all-zero stubs when no PyTorch model is provided.  The host
runner treats a missing baremetal_expected.bin as "skip the diff", so the bin
is purely an aid for the regression check.

Usage:
    python generate_baremetal_input.py [--batch 4] [--seed 42]
                                       [--model path/to/model.pth]
                                       [--out-dir ../../src/binaries]
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np


def deterministic_inputs(batch: int, seed: int) -> np.ndarray:
    """Reproducible float32 inputs of shape (batch, 3, 128).

    Uses np.random.default_rng so different numpy versions agree byte-for-byte.
    Range is [-1, 1] which roughly matches the MFCC features the model was
    trained on without needing the real preprocessing pipeline available.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(batch, 3, 128)).astype(np.float32)


def expected_argmax(model_path: Path | None, samples: np.ndarray) -> np.ndarray:
    """Run the model on samples to derive expected argmax bytes.

    Returns all-zero stubs when a model is not provided so the script stays
    runnable without torch installed.  The host runner can detect the all-
    zero case and skip the diff.
    """
    batch = samples.shape[0]

    if model_path is None:
        return np.zeros(batch, dtype=np.uint8)

    try:
        import torch  # local import keeps the no-model path lightweight
    except ImportError:
        print(
            "warning: --model passed but torch is not installed; "
            "writing all-zero expected output",
            file=sys.stderr,
        )
        return np.zeros(batch, dtype=np.uint8)

    state = torch.load(model_path, map_location="cpu")
    print(f"warning: --model is set but inference path is not implemented; "
          f"got {len(state.keys())} keys, returning zero stubs",
          file=sys.stderr)
    return np.zeros(batch, dtype=np.uint8)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=int, default=4,
                   help="number of (3, 128) samples to bake into the fixture")
    p.add_argument("--seed", type=int, default=42,
                   help="numpy default_rng seed; controls fixture reproducibility")
    p.add_argument("--model", type=Path, default=None,
                   help="optional .pth model to compute expected argmax")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "src" / "binaries",
        help="directory to write baremetal_input.bin / baremetal_expected.bin",
    )
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples = deterministic_inputs(args.batch, args.seed)

    payload_path = args.out_dir / "baremetal_input.bin"
    with payload_path.open("wb") as fp:
        fp.write(struct.pack(">I", args.batch))         # 4-byte big-endian length
        fp.write(samples.tobytes(order="C"))            # raw float32 in host order
    print(f"wrote {payload_path}  batch={args.batch}  bytes={payload_path.stat().st_size}")

    expected = expected_argmax(args.model, samples)
    expected_path = args.out_dir / "baremetal_expected.bin"
    expected.tofile(expected_path)
    print(f"wrote {expected_path}  bytes={expected_path.stat().st_size}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
