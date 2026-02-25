#!/usr/bin/env python3
"""
Constellation and eye diagram generator for CSOI RF signal data.

Reads CSOI .mat files and produces:
  - Constellation diagram: I vs Q scatter (16-QAM shows 16 clusters)
  - Eye diagram: I and Q levels overlaid across symbol periods

Uses same data source as waterfall_csoi.py.
"""

import argparse
import glob
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parent.parent / "Data" / "README (un-zipped includes COMMS signals of interest)" / "Comms Signals of Interest"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "signal_plots_output"


def load_csoi_mat(mat_path: Path) -> dict:
    """Load a CSOI .mat file and return IQ data plus metadata."""
    data = loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)

    iq = np.asarray(data["csoi_ComplexVoltageData"]).flatten()
    fs = float(np.squeeze(data["csoi_Fs"]))
    fc = float(np.squeeze(data.get("csoi_OffsetFrequency", 0)))
    sym_rate = float(np.squeeze(data.get("symRate", fs)))
    time_step_ns = data.get("csoi_TimeStep_ns")
    if time_step_ns is not None:
        time_step_ns = float(np.squeeze(time_step_ns))

    return {
        "iq": iq,
        "fs": fs,
        "fc": fc,
        "sym_rate": sym_rate,
        "time_step_ns": time_step_ns,
        "n_samples": len(iq),
    }


def plot_constellation(
    iq: np.ndarray,
    title: str,
    out_path: Path,
    max_points: int = 50000,
) -> None:
    """Plot I vs Q constellation diagram (16-QAM: 16 clusters on a 4x4 grid)."""
    iq = np.asarray(iq, dtype=np.complex128)
    I = np.real(iq)  # In-phase (x-axis)
    Q = np.imag(iq)  # Quadrature (y-axis)

    if len(I) > max_points:
        idx = np.random.default_rng(42).choice(len(I), max_points, replace=False)
        I, Q = I[idx], Q[idx]

    fig, ax = plt.subplots(figsize=(8, 8))
    # Use hexbin for density: 16-QAM clusters show as distinct bright regions
    hb = ax.hexbin(I, Q, gridsize=64, cmap="Blues", mincnt=1, edgecolors="none")
    fig.colorbar(hb, ax=ax, label="Count")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("In-phase (I) ← real part of signal")
    ax.set_ylabel("Quadrature (Q) ← imaginary part")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_eye_diagram(
    iq: np.ndarray,
    sym_rate: float,
    fs: float,
    title: str,
    out_path: Path,
    syms_per_trace: int = 3,
    n_traces: int = 500,
    max_samples: int = 10000000000,
) -> None:
    """
    Plot eye diagram: overlay I and Q over symbol-period windows.

    With 1 sample per symbol, overlays short segments to show discrete levels
    and transitions.
    """
    iq = np.asarray(iq, dtype=np.complex128)
    if len(iq) > max_samples:
        iq = iq[:max_samples]

    # Samples per symbol (typically 1 for this dataset)
    sps = max(1, int(round(fs / sym_rate)))
    I = np.real(iq)
    Q = np.imag(iq)

    # Use symbol-rate samples (every sps-th sample)
    I_sym = I[::sps]
    Q_sym = Q[::sps]

    n_syms = len(I_sym)
    if n_syms < syms_per_trace:
        return

    fig, (ax_i, ax_q) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    n_segments = min(n_traces, (n_syms - syms_per_trace) // 1)
    t = np.arange(syms_per_trace, dtype=float)

    for ax, data, label in [(ax_i, I_sym, "I"), (ax_q, Q_sym, "Q")]:
        for i in range(0, n_segments):
            start = i * (n_syms // max(1, n_segments))
            if start + syms_per_trace > n_syms:
                break
            seg = data[start : start + syms_per_trace]
            ax.plot(t, seg, color="steelblue", alpha=0.15, linewidth=0.8)
        ax.set_ylabel(f"{label} (V)")
        ax.set_title(f"{label} - Eye diagram ({syms_per_trace} symbols per trace)")
        ax.grid(True, alpha=0.3)
        ax.axvline(syms_per_trace / 2 - 0.5, color="red", linestyle="--", alpha=0.5)

    ax_q.set_xlabel("Symbol index")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_iq_levels(
    iq: np.ndarray,
    title: str,
    out_path: Path,
    n_symbols: int = 500,
) -> None:
    """Plot I and Q vs symbol index for a segment (shows discrete 16-QAM levels)."""
    iq = np.asarray(iq, dtype=np.complex128)
    I = np.real(iq[:n_symbols])
    Q = np.imag(iq[:n_symbols])

    fig, (ax_i, ax_q) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    x = np.arange(n_symbols)
    ax_i.step(x, I, where="mid", color="steelblue", linewidth=0.8)
    ax_i.set_ylabel("I")
    ax_i.set_title("In-phase (I) vs symbol index")
    ax_i.grid(True, alpha=0.3)

    ax_q.step(x, Q, where="mid", color="darkorange", linewidth=0.8)
    ax_q.set_ylabel("Q")
    ax_q.set_xlabel("Symbol index")
    ax_q.set_title("Quadrature (Q) vs symbol index")
    ax_q.grid(True, alpha=0.3)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def process_mat_file(
    mat_path: Path,
    output_dir: Path,
    max_samples: Optional[int] = None,
    max_constellation_points: int = 50000,
) -> None:
    """Load one .mat file and generate constellation + eye + IQ-levels plots."""
    print(f"Processing {mat_path.name}...")

    meta = load_csoi_mat(mat_path)
    iq = meta["iq"]
    fs = meta["fs"]
    fc = meta["fc"]
    sym_rate = meta.get("sym_rate", fs)

    if max_samples is not None and len(iq) > max_samples:
        iq = iq[:max_samples]
        print(f"  Truncated to {max_samples} samples")

    base = mat_path.stem
    title_base = f"{base}  (Fs={fs/1e9:.2f} GHz, Fc={fc/1e9:.2f} GHz)"

    # Constellation
    out_const = output_dir / f"{base}_constellation.png"
    plot_constellation(iq, f"Constellation: {title_base}", out_const, max_points=max_constellation_points)
    print(f"  Saved {out_const}")

    # Eye diagram
    out_eye = output_dir / f"{base}_eye.png"
    plot_eye_diagram(iq, sym_rate, fs, f"Eye: {title_base}", out_eye)
    print(f"  Saved {out_eye}")

    # I/Q levels
    out_levels = output_dir / f"{base}_iq_levels.png"
    plot_iq_levels(iq, f"I/Q Levels: {title_base}", out_levels)
    print(f"  Saved {out_levels}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate constellation and eye diagrams from CSOI .mat RF signal files."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing csoi_*_matfile.mat files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save plots",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100000,
        help="Max samples per file (default: 100000 for speed)",
    )
    parser.add_argument(
        "--max-constellation-points",
        type=int,
        default=50000,
        help="Max points in constellation scatter (default: 50000)",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Specific .mat files to process",
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"Data directory not found: {args.data_dir}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.files:
        mat_paths = [Path(p) for p in args.files if Path(p).exists()]
    else:
        pattern = args.data_dir / "csoi_*_matfile.mat"
        mat_paths = sorted(glob.glob(str(pattern)))
        mat_paths = [Path(p) for p in mat_paths]

    if not mat_paths:
        print(f"No .mat files found in {args.data_dir}")
        return

    print(f"Found {len(mat_paths)} .mat file(s)")
    for p in mat_paths:
        process_mat_file(
            p,
            args.output_dir,
            max_samples=args.max_samples,
            max_constellation_points=args.max_constellation_points,
        )
    print("Done.")


if __name__ == "__main__":
    main()
