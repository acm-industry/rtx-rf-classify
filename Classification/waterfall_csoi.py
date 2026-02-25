#!/usr/bin/env python3
"""
Waterfall plot generator for RF signal data in the Data folder.

Reads CSOI (Comms Signals of Interest) .mat files from:
  Data/README (un-zipped includes COMMS signals of interest)/Comms Signals of Interest/

Each .mat file contains:
  - csoi_ComplexVoltageData: complex IQ samples
  - csoi_Fs: sample rate (Hz)
  - csoi_OffsetFrequency: center frequency (Hz)
  - csoi_TimeStep_ns: time step (ns)

Waterfall format: time (rows) vs frequency (columns), color = power (dB).
"""

import argparse
import glob
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


# Default paths relative to script location
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parent.parent / "Data" / "README (un-zipped includes COMMS signals of interest)" / "Comms Signals of Interest"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "waterfall_output"


def load_csoi_mat(mat_path: Path) -> dict:
    """Load a CSOI .mat file and return IQ data plus metadata."""
    data = loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)

    iq = np.asarray(data["csoi_ComplexVoltageData"]).flatten()
    fs = float(np.squeeze(data["csoi_Fs"]))
    fc = float(np.squeeze(data.get("csoi_OffsetFrequency", 0)))
    # Optional: time step in ns
    time_step_ns = data.get("csoi_TimeStep_ns")
    if time_step_ns is not None:
        time_step_ns = float(np.squeeze(time_step_ns))

    return {
        "iq": iq,
        "fs": fs,
        "fc": fc,
        "time_step_ns": time_step_ns,
        "n_samples": len(iq),
    }


def compute_waterfall(
    iq: np.ndarray,
    fs: float,
    nfft: int = 1024,
    hop: Optional[int] = None,
    overlap_pct: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute waterfall (spectrogram) from complex IQ data.

    Returns:
        S: 2D power in dB, shape (n_time_bins, n_freq_bins)
        freqs: frequency axis (Hz), relative to center
        times: time axis (seconds)
    """
    if hop is None:
        hop = int(nfft * (1 - overlap_pct))

    window = np.hanning(nfft)

    n_segments = max(1, (len(iq) - nfft) // hop + 1)
    S = np.zeros((n_segments, nfft), dtype=np.float64)

    for i in range(n_segments):
        start = i * hop
        end = start + nfft
        if end > len(iq):
            break
        segment = np.asarray(iq[start:end], dtype=np.complex128) * window
        spec = np.fft.fft(segment)
        power = np.abs(spec) ** 2
        # fftshift to put DC at center (0 Hz)
        power = np.fft.fftshift(power)
        power = np.maximum(power, 1e-20)
        S[i, :] = 10 * np.log10(power)

    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / fs))
    times = np.arange(S.shape[0]) * hop / fs

    return S, freqs, times


def plot_waterfall(
    S: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    fc: float,
    title: str,
    out_path: Path,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """Save a single waterfall plot."""
    fig, ax = plt.subplots(figsize=(12, 8))

    if vmin is None:
        vmin = np.percentile(S, 5)
    if vmax is None:
        vmax = np.percentile(S, 98)

    extent = [
        freqs[0] * 1e-6,   # MHz
        freqs[-1] * 1e-6,
        times[0] * 1e6,    # µs
        times[-1] * 1e6,
    ]

    im = ax.imshow(
        S,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
    )

    ax.set_xlabel("Frequency offset (MHz)")
    ax.set_ylabel("Time (µs)")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, label="Power (dB)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def process_mat_file(
    mat_path: Path,
    output_dir: Path,
    nfft: int = 1024,
    overlap: float = 0.5,
    max_samples: Optional[int] = None,
) -> None:
    """Load one .mat file, compute waterfall, and save plot."""
    print(f"Processing {mat_path.name}...")

    meta = load_csoi_mat(mat_path)
    iq = meta["iq"]
    fs = meta["fs"]
    fc = meta["fc"]

    if max_samples is not None and len(iq) > max_samples:
        iq = iq[:max_samples]
        print(f"  Truncated to {max_samples} samples for faster plotting")

    S, freqs, times = compute_waterfall(iq, fs, nfft=nfft, overlap_pct=overlap)

    base = mat_path.stem
    title = f"{base}\nFs={fs/1e9:.2f} GHz, Fc={fc/1e9:.2f} GHz"
    out_path = output_dir / f"{base}_waterfall.png"

    plot_waterfall(S, freqs, times, fc, title, out_path)
    print(f"  Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate waterfall plots from CSOI .mat RF signal files."
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
        help="Directory to save waterfall images",
    )
    parser.add_argument(
        "--nfft",
        type=int,
        default=1024,
        help="FFT size for spectrogram",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Overlap fraction between segments (0–1)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per file (for quicker preview; default: use all)",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Specific .mat files to process (default: all in data-dir)",
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"Data directory not found: {args.data_dir}")
        print("  Pass --data-dir with the path to the Comms Signals of Interest folder.")
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
            nfft=args.nfft,
            overlap=args.overlap,
            max_samples=args.max_samples,
        )
    print("Done.")


if __name__ == "__main__":
    main()
