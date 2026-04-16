from __future__ import annotations

import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import loadmat


@dataclass(frozen=True)
class MidasHeader:
    version: str
    head_rep: str
    data_rep: str
    data_start: float
    data_size_bytes: float
    file_type: int
    data_format: str
    xstart: float
    xdelta: float
    xunits: int
    raster_size: int
    ystart: float
    ydelta: float
    yunits: int

    @property
    def is_complex(self) -> bool:
        return self.data_format.startswith("C")

    @property
    def sample_rate_hz(self) -> float:
        return 1.0 / self.xdelta if self.xdelta else math.nan


@dataclass(frozen=True)
class ScenarioCapture:
    path: Path
    rx_index: int
    channel_index: int
    rx_name: str
    rx_position: tuple[float, float]
    with_interferers: bool


MIDAS_CAPTURE_RE = re.compile(
    r"^(?:no_interferers_)?scenario_A_1_(?P<rx>\d+)_(?P<channel>\d+)_(?P<frame>\d+)\.tmp$"
)


def load_csoi_mat(mat_path: str | Path) -> dict:
    """Load one CSOI .mat file and return the raw IQ plus metadata."""
    mat_path = Path(mat_path)
    data = loadmat(mat_path)
    signal_keys = [k for k in data.keys() if k.endswith("_ComplexVoltageData")]
    if len(signal_keys) != 1:
        raise ValueError(f"Could not infer signal key from {mat_path}")

    iq_key = signal_keys[0]
    prefix = iq_key[: -len("_ComplexVoltageData")]

    fs_key = f"{prefix}_Fs"
    offset_key = f"{prefix}_OffsetFrequency"

    iq = np.asarray(data[iq_key]).reshape(-1).astype(np.complex64)
    return {
        "path": mat_path,
        "signal_name": prefix,
        "iq": iq,
        "sample_rate_hz": float(np.asarray(data[fs_key]).squeeze()),
        "offset_frequency_hz": float(np.asarray(data[offset_key]).squeeze()),
        "mat_keys": sorted(k for k in data.keys() if not k.startswith("__")),
    }


def read_midas_header(path: str | Path) -> MidasHeader:
    """Parse the fixed MIDAS Blue header used by the scenario .tmp files."""
    path = Path(path)
    with path.open("rb") as handle:
        first_512 = handle.read(512)

    version = first_512[:4].decode("ascii")
    head_rep = first_512[4:8].decode("ascii")
    data_rep = first_512[8:12].decode("ascii")
    endian = "<" if head_rep == "EEEI" else ">"

    detached, write_protect, pipe, ext_start, ext_size = struct.unpack(
        f"{endian}5i", first_512[12:32]
    )
    del detached, write_protect, pipe, ext_start, ext_size

    data_start, data_size, file_type = struct.unpack(f"{endian}2di", first_512[32:52])
    data_format = first_512[52:54].decode("ascii")

    xstart, xdelta, xunits = struct.unpack(f"{endian}ddi", first_512[256:276])
    raster_size = struct.unpack(f"{endian}i", first_512[276:280])[0]
    ystart, ydelta, yunits = struct.unpack(f"{endian}ddi", first_512[280:300])

    if file_type < 2000:
        raster_size = 1

    return MidasHeader(
        version=version,
        head_rep=head_rep,
        data_rep=data_rep,
        data_start=data_start,
        data_size_bytes=data_size,
        file_type=file_type,
        data_format=data_format,
        xstart=xstart,
        xdelta=xdelta,
        xunits=xunits,
        raster_size=raster_size,
        ystart=ystart,
        ydelta=ydelta,
        yunits=yunits,
    )


def load_midas_iq(path: str | Path) -> tuple[np.ndarray, MidasHeader]:
    """Load a scenario .tmp file as complex IQ."""
    path = Path(path)
    header = read_midas_header(path)

    if header.data_format != "CI":
        raise ValueError(f"Unsupported MIDAS format {header.data_format!r} in {path}")

    with path.open("rb") as handle:
        handle.seek(int(header.data_start))
        count = int(header.data_size_bytes) // np.dtype("<i2").itemsize
        raw = np.fromfile(handle, dtype="<i2", count=count)

    if raw.size % 2:
        raise ValueError(f"Expected I/Q pairs in {path}, got odd sample count")

    iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
    return iq, header


def complex_to_2channel(iq: np.ndarray) -> np.ndarray:
    """Convert complex IQ to a CNN-friendly 2 x N float array."""
    iq = np.asarray(iq)
    return np.stack([iq.real, iq.imag], axis=0).astype(np.float32)


def make_windows(
    iq: np.ndarray,
    window_size: int,
    stride: int | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Split one complex capture into windows shaped (num_windows, 2, window_size)."""
    if stride is None:
        stride = window_size

    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")

    iq = np.asarray(iq).reshape(-1)
    frames = []
    for start in range(0, iq.size - window_size + 1, stride):
        frame = complex_to_2channel(iq[start : start + window_size])
        if normalize:
            scale = np.max(np.abs(frame))
            if scale > 0:
                frame = frame / scale
        frames.append(frame)

    if not frames:
        return np.empty((0, 2, window_size), dtype=np.float32)

    return np.stack(frames, axis=0)


def load_scenario_metadata(json_path: str | Path) -> dict:
    json_path = Path(json_path)
    return json.loads(json_path.read_text())


def list_scenario_a1_captures(
    iq_dir: str | Path,
    scenario_json: str | Path,
    with_interferers: bool = True,
) -> list[ScenarioCapture]:
    """Match scenario capture filenames to RxVehicle metadata."""
    iq_dir = Path(iq_dir)
    scenario = load_scenario_metadata(scenario_json)
    rx_players = [p for p in scenario["Players"] if p["name"].startswith("RxVehicle_")]

    captures = []
    for path in sorted(iq_dir.glob("*.tmp")):
        match = MIDAS_CAPTURE_RE.match(path.name)
        if not match:
            continue

        rx_index = int(match.group("rx"))
        channel_index = int(match.group("channel"))
        frame_index = int(match.group("frame"))
        if frame_index != 0:
            continue

        rx_player = rx_players[rx_index]
        east = float(rx_player["geometryObject"]["offsetFromArrayCenter"]["E"])
        down = float(rx_player["geometryObject"]["offsetFromArrayCenter"]["D"])
        captures.append(
            ScenarioCapture(
                path=path,
                rx_index=rx_index,
                channel_index=channel_index,
                rx_name=rx_player["name"],
                rx_position=(east, down),
                with_interferers=with_interferers,
            )
        )

    return captures


def build_scenario_windows(
    captures: Iterable[ScenarioCapture],
    window_size: int,
    stride: int | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Load many captures and return CNN-ready windows plus per-window metadata."""
    examples = []
    metadata = []

    for capture in captures:
        iq, header = load_midas_iq(capture.path)
        windows = make_windows(iq, window_size=window_size, stride=stride)
        examples.append(windows)
        metadata.extend(
            {
                "path": str(capture.path),
                "rx_index": capture.rx_index,
                "channel_index": capture.channel_index,
                "rx_name": capture.rx_name,
                "rx_position": capture.rx_position,
                "with_interferers": capture.with_interferers,
                "sample_rate_hz": header.sample_rate_hz,
            }
            for _ in range(windows.shape[0])
        )

    if not examples:
        return np.empty((0, 2, window_size), dtype=np.float32), []

    return np.concatenate(examples, axis=0), metadata


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[3]

    csoi_file = repo_root / "csoi" / "csoi_0_matfile.mat"
    csoi = load_csoi_mat(csoi_file)
    print(f"Loaded {csoi['signal_name']} with {csoi['iq'].shape[0]:,} complex samples")
    print(f"Sample rate: {csoi['sample_rate_hz'] / 1e6:.2f} MHz")

    scenario_root = repo_root / "Scenario A_1 (easy)"
    captures = list_scenario_a1_captures(
        scenario_root / "IQ",
        scenario_root / "scenario_A_1.json",
        with_interferers=True,
    )
    print(f"Scenario A_1 captures: {len(captures)}")

    sample_iq, sample_header = load_midas_iq(captures[0].path)
    print(f"First scenario capture shape: {sample_iq.shape[0]:,} complex samples")
    print(f"Header format: {sample_header.data_format}, Fs ~= {sample_header.sample_rate_hz / 1e9:.3f} GHz")

    windows = make_windows(sample_iq, window_size=1024, stride=512)
    print(f"CNN-ready windows from one capture: {windows.shape}")
