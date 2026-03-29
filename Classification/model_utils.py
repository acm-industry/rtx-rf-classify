"""
Load RadioML (or compatible) checkpoints with FP32 or FP16 weights interchangeably.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union

import torch
import torch.nn as nn

try:
    from .cnn import RadioMLCNN
except ImportError:
    from cnn import RadioMLCNN

Precision = Literal["auto", "fp32", "fp16"]
CheckpointPath = Union[str, Path]


@dataclass(frozen=True)
class LoadedModel:
    """Result of load_checkpoint_into_model: model placement and dtype for inputs."""

    model: nn.Module
    param_dtype: torch.dtype

    def match_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Move inputs to the model device and dtype (for FP16/FP32 inference)."""
        dev = next(self.model.parameters()).device
        return x.to(device=dev, dtype=self.param_dtype)


def _extract_state_dict(payload: object) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        if "state_dict" in payload:
            inner = payload["state_dict"]
            if isinstance(inner, dict):
                return inner
        if "model" in payload and isinstance(payload["model"], dict):
            inner = payload["model"]
            if isinstance(inner, dict):
                return inner
        if payload and all(isinstance(v, torch.Tensor) for v in payload.values()):
            return payload  # type: ignore[return-value]
    raise ValueError(
        "Unsupported checkpoint format: expected a state_dict or a dict with 'state_dict' / 'model' key"
    )


def load_checkpoint_into_model(
    model: nn.Module,
    path: CheckpointPath,
    device: torch.device | str | None = None,
    precision: Precision = "auto",
    *,
    strict: bool = True,
) -> LoadedModel:
    """
    Load weights from ``path`` into ``model``, then apply ``precision`` and move to ``device``.

    - ``auto``: keep dtype stored in the checkpoint (FP32 or FP16).
    - ``fp32``: ``model.float()`` after load.
    - ``fp16``: ``model.half()`` after load.

    For inference, call ``LoadedModel.match_inputs(batch)`` so activations match parameter dtype.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)

    state = _extract_state_dict(payload)
    checkpoint_is_fp16 = any(
        isinstance(v, torch.Tensor) and v.is_floating_point() and v.dtype == torch.float16
        for v in state.values()
    )

    model.load_state_dict(state, strict=strict)

    # Modules default to float32; load_state_dict may promote FP16 tensors into FP32
    # parameters. For FP16 checkpoints, explicitly convert the module so inference
    # matches the saved precision.
    if precision == "fp32":
        model.float()
    elif precision == "fp16":
        model.half()
    else:  # auto
        model.half() if checkpoint_is_fp16 else model.float()

    model.to(device)
    param_dtype = next(model.parameters()).dtype
    return LoadedModel(model=model, param_dtype=param_dtype)


def load_radioml_cnn(
    path: CheckpointPath,
    num_classes: int = 11,
    dropout_rate: float = 0.5,
    device: torch.device | str | None = None,
    precision: Precision = "auto",
    *,
    strict: bool = True,
) -> LoadedModel:
    """Construct :class:`RadioMLCNN` and load ``path`` (convenience wrapper)."""
    model = RadioMLCNN(num_classes=num_classes, dropout_rate=dropout_rate)
    return load_checkpoint_into_model(model, path, device=device, precision=precision, strict=strict)
