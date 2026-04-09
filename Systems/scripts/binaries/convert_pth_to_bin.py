#!/usr/bin/env python3
"""
Export embedded weight .bin files for Systems (see docs/weights_layout.md).

Default: IEEE float32 per scalar. Use --fp16 for half the on-disk size; the C++ binary
must then be built with -DWEIGHTS_FP16=1 (CMake option WEIGHTS_FP16=ON).
"""

from __future__ import annotations

import argparse

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="PyTorch state_dict .pth (Conv1d features.* + classifier.*)")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Write binary16 weights (~half the bytes of float32). Requires WEIGHTS_FP16 in C++ build.",
    )
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu")
    print(state.keys())

    np_dtype = np.float16 if args.fp16 else np.float32

    def save_tensor(tensor: torch.Tensor, filename: str) -> None:
        arr = tensor.detach().cpu().numpy().astype(np_dtype)
        arr.tofile(filename)
        print(f"wrote {filename}  shape={arr.shape}  dtype={arr.dtype}")

    def save_bn(prefix: str, layer_idx: int) -> None:
        gamma = state[f"features.{layer_idx}.weight"]
        beta = state[f"features.{layer_idx}.bias"]
        mean = state[f"features.{layer_idx}.running_mean"]
        var = state[f"features.{layer_idx}.running_var"]

        packed = np.concatenate(
            [gamma.numpy(), beta.numpy(), mean.numpy(), var.numpy()]
        ).astype(np_dtype)

        packed.tofile(prefix)
        print(f"wrote {prefix}  shape={(4, gamma.numel())}  dtype={packed.dtype}")

    # Layer 1
    save_tensor(state["features.0.weight"], "weights_conv_1d_l1.bin")
    save_tensor(state["features.0.bias"], "weights_conv_1d_bias_l1.bin")
    save_bn("weights_bn_1d_l1.bin", 1)

    # Layer 2
    save_tensor(state["features.4.weight"], "weights_conv_1d_l2.bin")
    save_tensor(state["features.4.bias"], "weights_conv_1d_bias_l2.bin")
    save_bn("weights_bn_1d_l2.bin", 5)

    # Layer 3
    save_tensor(state["features.8.weight"], "weights_conv_1d_l3.bin")
    save_tensor(state["features.8.bias"], "weights_conv_1d_bias_l3.bin")
    save_bn("weights_bn_1d_l3.bin", 9)

    # Layer 4
    save_tensor(state["features.11.weight"], "weights_conv_1d_l4.bin")
    save_tensor(state["features.11.bias"], "weights_conv_1d_bias_l4.bin")
    save_bn("weights_bn_1d_l4.bin", 12)

    # ---- Linear Layers ----

    save_tensor(state["classifier.3.weight"], "linear_mat_1.bin")
    save_tensor(state["classifier.3.bias"], "linear_add_1.bin")

    save_tensor(state["classifier.6.weight"], "linear_mat_2.bin")
    save_tensor(state["classifier.6.bias"], "linear_add_2.bin")

    print("\nDone.")


if __name__ == "__main__":
    main()
