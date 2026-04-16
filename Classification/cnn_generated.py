from __future__ import annotations

import argparse
import copy
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a stronger RadioML CNN baseline.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("RML2016.10a_dict.pkl"),
        help="Path to the RadioML pickle file.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2016)
    parser.add_argument(
        "--min-snr",
        type=int,
        default=None,
        help="Optional SNR floor. Example: use --min-snr 0 to train/evaluate on easier samples only.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=12,
        help="Early stopping patience in epochs.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("radioml_cnn_generated.pth"),
        help="Output model checkpoint path.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_radioml(data_path: Path, min_snr: int | None = None):
    with data_path.open("rb") as handle:
        unpickler = pickle._Unpickler(handle)
        unpickler.encoding = "latin1"
        data_dict = unpickler.load()

    snrs, mods = map(
        lambda idx: sorted({key[idx] for key in data_dict.keys()}),
        [1, 0],
    )

    X_parts = []
    mod_labels = []
    snr_labels = []
    for mod in mods:
        for snr in snrs:
            if min_snr is not None and snr < min_snr:
                continue
            samples = data_dict[(mod, snr)].astype(np.float32)
            X_parts.append(samples)
            mod_labels.extend([mods.index(mod)] * samples.shape[0])
            snr_labels.extend([snr] * samples.shape[0])

    X = np.vstack(X_parts)
    y = np.asarray(mod_labels, dtype=np.int64)
    snr = np.asarray(snr_labels, dtype=np.int64)
    return X, y, snr, mods


def stratified_split_indices(
    y: np.ndarray,
    snr: np.ndarray,
    val_frac: float,
    test_frac: float,
    seed: int,
):
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []
    test_idx = []

    for cls in np.unique(y):
        for snr_value in np.unique(snr):
            indices = np.where((y == cls) & (snr == snr_value))[0]
            if indices.size == 0:
                continue
            rng.shuffle(indices)

            n_test = int(round(indices.size * test_frac))
            n_val = int(round(indices.size * val_frac))
            n_test = min(n_test, indices.size)
            n_val = min(n_val, indices.size - n_test)

            test_slice = indices[:n_test]
            val_slice = indices[n_test : n_test + n_val]
            train_slice = indices[n_test + n_val :]

            train_idx.extend(train_slice.tolist())
            val_idx.extend(val_slice.tolist())
            test_idx.extend(test_slice.tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def normalize_per_example(x: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.mean(np.square(x), axis=(1, 2), keepdims=True) + 1e-8)
    return x / scale


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x = torch.relu(x + residual)
        return x


class RadioMLResNet(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.25):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.layers = nn.Sequential(
            ResidualBlock1D(64, 64, stride=1, dropout=dropout),
            ResidualBlock1D(64, 128, stride=2, dropout=dropout),
            ResidualBlock1D(128, 128, stride=1, dropout=dropout),
            ResidualBlock1D(128, 256, stride=2, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layers(x)
        x = self.pool(x)
        return self.head(x)


def build_loaders(
    X: np.ndarray,
    y: np.ndarray,
    snr: np.ndarray,
    args: argparse.Namespace,
):
    X = normalize_per_example(X)
    train_idx, val_idx, test_idx = stratified_split_indices(
        y=y,
        snr=snr,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    train_dataset = TensorDataset(
        torch.from_numpy(X[train_idx]),
        torch.from_numpy(y[train_idx]),
        torch.from_numpy(snr[train_idx]),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X[val_idx]),
        torch.from_numpy(y[val_idx]),
        torch.from_numpy(snr[val_idx]),
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X[test_idx]),
        torch.from_numpy(y[test_idx]),
        torch.from_numpy(snr[test_idx]),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader, test_loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None = None,
):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for inputs, labels, _snr in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(inputs)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += labels.size(0)

    return total_loss / total_examples, total_correct / total_examples


def evaluate_by_snr(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[int, float]:
    model.eval()
    correct_by_snr: dict[int, int] = {}
    total_by_snr: dict[int, int] = {}

    with torch.no_grad():
        for inputs, labels, snr in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(inputs)
            preds = logits.argmax(dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()
            snr_np = snr.numpy()

            for pred, target, snr_value in zip(preds, labels_np, snr_np):
                snr_value = int(snr_value)
                total_by_snr[snr_value] = total_by_snr.get(snr_value, 0) + 1
                correct_by_snr[snr_value] = correct_by_snr.get(snr_value, 0) + int(pred == target)

    return {
        snr_value: correct_by_snr[snr_value] / total_by_snr[snr_value]
        for snr_value in sorted(total_by_snr)
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X, y, snr, mods = load_radioml(args.data_path, min_snr=args.min_snr)
    print(f"Loaded {X.shape[0]:,} examples across {len(mods)} classes")
    if args.min_snr is not None:
        print(f"Using SNR >= {args.min_snr} dB")

    train_loader, val_loader, test_loader = build_loaders(X, y, snr, args)
    print(
        f"Split sizes: train={len(train_loader.dataset):,}, "
        f"val={len(val_loader.dataset):,}, test={len(test_loader.dataset):,}"
    )

    model = RadioMLResNet(num_classes=len(mods), dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    best_state = None
    best_val_loss = float("inf")
    best_val_acc = 0.0
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4%} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4%} | "
            f"lr={current_lr:.2e}"
        )

        if stale_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    model.load_state_dict(best_state)
    test_loss, test_acc = run_epoch(model, test_loader, criterion, device)
    snr_acc = evaluate_by_snr(model, test_loader, device)

    print(f"Best val accuracy: {best_val_acc:.4%}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4%}")
    print("Accuracy by SNR:")
    for snr_value, acc in snr_acc.items():
        print(f"  {snr_value:>3} dB: {acc:.4%}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": mods,
            "args": vars(args),
            "test_accuracy": test_acc,
            "snr_accuracy": snr_acc,
        },
        args.save_path,
    )
    print(f"Saved checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
