#!/usr/bin/env python3
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-time conversion: radioml_cnn_pytorch.pth -> radioml_cnn_weights.bin"
    )
    parser.add_argument("--pth", type=Path, required=True, help="Path to .pth state_dict")
    parser.add_argument("--out", type=Path, required=True, help="Output .bin path")
    args = parser.parse_args()

    sd = torch.load(args.pth, map_location="cpu")
    keys = (
        "conv1.weight",
        "conv1.bias",
        "conv2.weight",
        "conv2.bias",
        "fc1.weight",
        "fc1.bias",
        "fc2.weight",
        "fc2.bias",
    )
    expected_shapes = {
        "conv1.weight": (256, 1, 1, 3),
        "conv1.bias": (256,),
        "conv2.weight": (80, 256, 2, 3),
        "conv2.bias": (80,),
        "fc1.weight": (256, 10400),
        "fc1.bias": (256,),
        "fc2.weight": (11, 256),
        "fc2.bias": (11,),
    }

    for k in keys:
        if k not in sd:
            raise KeyError(f"Missing key: {k}")
        if tuple(sd[k].shape) != expected_shapes[k]:
            raise ValueError(f"Shape mismatch for {k}: got {tuple(sd[k].shape)} expected {expected_shapes[k]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        f.write(b"RMLW")
        f.write(struct.pack("<I", 1))
        for k in keys:
            a = sd[k].detach().cpu().contiguous().to(torch.float32).numpy()
            f.write(a.tobytes(order="C"))

    print(args.out)


if __name__ == "__main__":
    main()
