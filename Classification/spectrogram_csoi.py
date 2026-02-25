#!/usr/bin/env python3
"""
Spectrogram image generator for CSOI RF signal data.

Reads CSOI .mat files and produces spectrogram images:
  time (x-axis) vs frequency (y-axis), color = power (dB).

Same data source as waterfall_csoi.py and constellation_eye_csoi.py.
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
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "spectrogram_output"


def load_csoi_mat(mat_path: Path) -> dict:
    """Load a CSOI .mat file and return IQ data plus metadata."""
    data = loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)

    iq = np.asarray(data["csoi_ComplexVoltageData"]).flatten()
    fs = float(np.squeeze(data["csoi_Fs"]))
    fc = float(np.squeeze(data.get("csoi_OffsetFrequency", 0)))
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


def compute_spectrogram(
    iq: np.ndarray,
    fs: float,
    nfft: int = 1024,
    overlap_pct: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute spectrogram from complex IQ data (baseband, fftshifted).

    Returns:
        S_db: 2D power in dB, shape (n_freq_bins, n_time_bins)
        freqs: frequency axis (Hz), 0 Hz at center
        times: time axis (seconds)
    """
    nperseg = nfft
    hop = int(nperseg * (1 - overlap_pct))
    window = np.hanning(nperseg)

    n_segments = max(1, (len(iq) - nperseg) // hop + 1)
    S_full = np.zeros((nperseg, n_segments), dtype=np.float64)

    for i in range(n_segments):
        start = i * hop
        end = start + nperseg
        if end > len(iq):
            break
        seg = np.asarray(iq[start:end], dtype=np.complex128) * window
        spec = np.fft.fft(seg)
        power = np.abs(spec) ** 2
        power = np.maximum(power, 1e-20)
        S_full[:, i] = 10 * np.log10(power)

    S_db = np.fft.fftshift(S_full, axes=0)
    freqs = np.fft.fftshift(np.fft.fftfreq(nperseg, 1 / fs))
    times = np.arange(S_db.shape[1]) * hop / fs

    return S_db, freqs, times


def plot_spectrogram(
    S_db: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    title: str,
    out_path: Path,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """Save spectrogram as image (time x frequency, power in dB)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    if vmin is None:
        vmin = np.percentile(S_db, 5)
    if vmax is None:
        vmax = np.percentile(S_db, 98)

    extent = [
        times[0] * 1e6,   # µs
        times[-1] * 1e6,
        freqs[0] * 1e-6,  # MHz
        freqs[-1] * 1e-6,
    ]

    im = ax.imshow(
        S_db,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
    )

    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Frequency offset (MHz)")
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
    """Load one .mat file, compute spectrogram, save image."""
    print(f"Processing {mat_path.name}...")

    meta = load_csoi_mat(mat_path)
    iq = meta["iq"]
    fs = meta["fs"]
    fc = meta["fc"]

    if max_samples is not None and len(iq) > max_samples:
        iq = iq[:max_samples]
        print(f"  Truncated to {max_samples} samples")

    S_db, freqs, times = compute_spectrogram(iq, fs, nfft=nfft, overlap_pct=overlap)

    base = mat_path.stem
    title = f"Spectrogram: {base}  (Fs={fs/1e9:.2f} GHz, Fc={fc/1e9:.2f} GHz)"
    out_path = output_dir / f"{base}_spectrogram.png"

    plot_spectrogram(S_db, freqs, times, title, out_path)
    print(f"  Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate spectrogram images from CSOI .mat RF signal files."
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
        help="Directory to save spectrogram images",
    )
    parser.add_argument(
        "--nfft",
        type=int,
        default=1024,
        help="FFT size",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Overlap fraction (0–1)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per file (for faster preview)",
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
            nfft=args.nfft,
            overlap=args.overlap,
            max_samples=args.max_samples,
        )
    print("Done.")


if __name__ == "__main__":
    main()
