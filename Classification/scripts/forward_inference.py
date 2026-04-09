#!/usr/bin/env python3
"""
Run RadioML forward inference in forced FP32 or FP16 and report accuracy/timing.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Classification.model_utils import load_radioml_cnn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to .npz dataset containing X and optionally y",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "Systems" / "src" / "cnn" / "radioml_cnn_pytorch.pth",
        help="Checkpoint to load (default: Systems/src/cnn/radioml_cnn_pytorch.pth)",
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16"),
        required=True,
        help="Force model weights and activations to run in fp32 or fp16",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Inference batch size",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of samples to evaluate",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=1,
        help="Number of warmup batches before timing",
    )
    parser.add_argument(
        "--save-preds",
        type=Path,
        default=None,
        help="Optional .npy path to save predicted class indices",
    )
    parser.add_argument(
        "--save-logits",
        type=Path,
        default=None,
        help="Optional .npy path to save model logits",
    )
    return parser.parse_args()


def prepare_inputs(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[1:] == (2, 128):
        x = x[:, None, :, :]
    elif x.ndim == 4 and x.shape[1:] == (1, 2, 128):
        pass
    else:
        raise ValueError(
            f"Unsupported X shape {x.shape}. Expected (N, 2, 128) or (N, 1, 2, 128)."
        )
    return np.ascontiguousarray(x.astype(np.float32, copy=False))


def prepare_labels(y: np.ndarray | None, n_samples: int) -> np.ndarray | None:
    if y is None:
        return None
    y = np.asarray(y)
    if y.ndim == 2:
        y = np.argmax(y, axis=1)
    elif y.ndim != 1:
        raise ValueError(f"Unsupported y shape {y.shape}. Expected (N,) or one-hot (N, C).")
    if y.shape[0] != n_samples:
        raise ValueError(f"X has {n_samples} samples but y has {y.shape[0]}.")
    return y.astype(np.int64, copy=False)


def main() -> None:
    args = parse_args()

    data = np.load(args.dataset)
    if "X" not in data:
        raise KeyError(f"{args.dataset} does not contain 'X'")
    x = prepare_inputs(data["X"])
    if args.limit is not None:
        x = x[: args.limit]
    y = prepare_labels(data["y"] if "y" in data else None, x.shape[0])
    if args.limit is not None and y is not None:
        y = y[: args.limit]

    loaded = load_radioml_cnn(
        args.checkpoint,
        num_classes=11,
        device=args.device,
        precision=args.precision,
    )
    model = loaded.model.eval()

    logits_parts: list[np.ndarray] = []
    preds_parts: list[np.ndarray] = []

    total_samples = x.shape[0]
    warmup_batches = max(args.warmup_batches, 0)
    n_batches = (total_samples + args.batch_size - 1) // args.batch_size

    with torch.inference_mode():
        for batch_idx in range(min(warmup_batches, n_batches)):
            start = batch_idx * args.batch_size
            end = min(start + args.batch_size, total_samples)
            batch = torch.from_numpy(x[start:end])
            _ = model(loaded.match_inputs(batch))

        if args.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        for start in range(0, total_samples, args.batch_size):
            end = min(start + args.batch_size, total_samples)
            batch = torch.from_numpy(x[start:end])
            out = model(loaded.match_inputs(batch))
            logits = out.float().cpu().numpy()
            preds = np.argmax(logits, axis=1).astype(np.uint8, copy=False)
            logits_parts.append(logits)
            preds_parts.append(preds)

        if args.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    logits_all = np.concatenate(logits_parts, axis=0) if logits_parts else np.empty((0, 11), dtype=np.float32)
    preds_all = np.concatenate(preds_parts, axis=0) if preds_parts else np.empty((0,), dtype=np.uint8)

    accuracy = None
    if y is not None:
        accuracy = float(np.mean(preds_all == y))

    print(f"precision={args.precision}")
    print(f"device={args.device}")
    print(f"checkpoint={args.checkpoint}")
    print(f"dataset={args.dataset}")
    print(f"samples={total_samples}")
    print(f"model_param_dtype={next(model.parameters()).dtype}")
    print(f"input_dtype={loaded.param_dtype}")
    print(f"elapsed_sec={elapsed:.6f}")
    print(f"samples_per_sec={total_samples / elapsed:.2f}" if elapsed > 0 else "samples_per_sec=inf")
    if accuracy is not None:
        print(f"accuracy={accuracy:.6f}")

    if args.save_preds is not None:
        args.save_preds.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_preds, preds_all)
        print(f"saved_preds={args.save_preds}")

    if args.save_logits is not None:
        args.save_logits.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_logits, logits_all)
        print(f"saved_logits={args.save_logits}")


if __name__ == "__main__":
    main()
