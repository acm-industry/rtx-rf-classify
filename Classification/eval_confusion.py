from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from cnn_generated import (
    RadioMLResNet,
    build_loaders,
    load_radioml,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a forward pass on the RadioML test split and save a confusion matrix.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("radioml_cnn_generated.pth"),
        help="Path to a checkpoint created by cnn_generated.py",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("RML2016.10a_dict.pkl"),
        help="Path to the RadioML pickle file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional override for evaluation batch size. Defaults to the checkpoint value or 512.",
    )
    parser.add_argument(
        "--save-prefix",
        type=Path,
        default=Path("radioml_confusion"),
        help="Prefix for saved confusion matrix artifacts.",
    )
    return parser.parse_args()


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def save_confusion_csv(cm: np.ndarray, classes: list[str], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *classes])
        for class_name, row in zip(classes, cm):
            writer.writerow([class_name, *row.tolist()])


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    ckpt_args = checkpoint.get("args", {})
    classes = checkpoint["classes"]

    seed = int(ckpt_args.get("seed", 2016))
    set_seed(seed)

    class EvalArgs:
        pass

    eval_args = EvalArgs()
    eval_args.batch_size = args.batch_size or int(ckpt_args.get("batch_size", 512))
    eval_args.val_frac = float(ckpt_args.get("val_frac", 0.15))
    eval_args.test_frac = float(ckpt_args.get("test_frac", 0.15))
    eval_args.seed = seed

    min_snr = ckpt_args.get("min_snr", None)

    X, y, snr, mods = load_radioml(args.data_path, min_snr=min_snr)
    if list(mods) != list(classes):
        raise ValueError("Checkpoint class ordering does not match dataset class ordering.")

    _train_loader, _val_loader, test_loader = build_loaders(X, y, snr, eval_args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RadioMLResNet(
        num_classes=len(classes),
        dropout=float(ckpt_args.get("dropout", 0.25)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, labels, _snr in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            logits = model(inputs)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(labels.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    cm = confusion_matrix(y_true, y_pred, num_classes=len(classes))
    accuracy = float((y_true == y_pred).mean())

    print(f"Device: {device}")
    print(f"Test examples: {y_true.shape[0]:,}")
    print(f"Accuracy: {accuracy:.4%}")
    print("Classes:")
    print(classes)
    print("Confusion matrix:")
    print(cm)

    npy_path = args.save_prefix.with_suffix(".npy")
    csv_path = args.save_prefix.with_suffix(".csv")
    np.save(npy_path, cm)
    save_confusion_csv(cm, classes, csv_path)
    print(f"Saved {npy_path}")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
