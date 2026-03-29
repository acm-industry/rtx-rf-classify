#!/usr/bin/env python3
"""
Convert a PyTorch checkpoint to float16 for all floating-point tensors in the model weights.

Writes a new file; never overwrites the input path. Intended for state_dict checkpoints
(e.g. from torch.save(model.state_dict(), path)).

If the file is a training checkpoint with 'state_dict' and 'optimizer' keys, only tensors
inside 'state_dict' (or 'model') are converted; optimizer and other keys are copied unchanged
so FP32 training resumes stay valid if you only use the nested state_dict for inference.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def _convert_float_tensors(obj: object) -> object:
    if isinstance(obj, torch.Tensor):
        return obj.half() if obj.is_floating_point() else obj
    if isinstance(obj, dict):
        return {k: _convert_float_tensors(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = [_convert_float_tensors(x) for x in obj]
        return type(obj)(t)
    return obj


def convert_checkpoint_payload(payload: object) -> object:
    """
    Convert floating tensors to float16.

    - Plain ``state_dict``: convert all float tensors.
    - Training checkpoint: if ``state_dict`` or ``model`` is present, only that mapping
      is converted; ``optimizer`` and other keys are left unchanged.
    """
    if not isinstance(payload, dict):
        return _convert_float_tensors(payload)

    if "state_dict" in payload and isinstance(payload["state_dict"], dict):
        out = dict(payload)
        out["state_dict"] = _convert_float_tensors(payload["state_dict"])
        return out
    if "model" in payload and isinstance(payload["model"], dict):
        out = dict(payload)
        out["model"] = _convert_float_tensors(payload["model"])
        return out

    return _convert_float_tensors(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="Input .pth file (state_dict or training checkpoint)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output path for FP16 checkpoint (must differ from input)",
    )
    args = parser.parse_args()

    inp = args.input.expanduser().resolve()
    out = args.output.expanduser().resolve()

    if not inp.is_file():
        print(f"error: input not found: {inp}", file=sys.stderr)
        sys.exit(1)
    if inp == out:
        print("error: --output must differ from input (refusing to overwrite in place)", file=sys.stderr)
        sys.exit(1)
    if out.exists():
        print(f"error: output already exists: {out} (delete or pick another path)", file=sys.stderr)
        sys.exit(1)

    try:
        payload = torch.load(inp, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(inp, map_location="cpu")

    converted = convert_checkpoint_payload(payload)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, out)
    print(f"Wrote FP16 checkpoint to {out}")


if __name__ == "__main__":
    main()
