"""Core image data structure and sRGB transfer functions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

# ── sRGB transfer functions (IEC 61966-2-1) ──


def linear_to_srgb(data: np.ndarray) -> np.ndarray:
    """Linear light → sRGB gamma. Handles values > 1.0 for HDR data."""
    data = np.maximum(data, 0.0)
    low = data * 12.92
    high = 1.055 * np.power(data, 1.0 / 2.4) - 0.055
    return np.where(data <= 0.0031308, low, high).astype(np.float32)


def srgb_to_linear(data: np.ndarray) -> np.ndarray:
    """sRGB gamma → linear light."""
    data = np.maximum(data, 0.0)
    low = data / 12.92
    high = np.power((data + 0.055) / 1.055, 2.4)
    return np.where(data <= 0.04045, low, high).astype(np.float32)


# ── Data structures ──


class Encoding(Enum):
    LINEAR = "linear"  # Linear light, values may exceed 1.0
    SRGB = "srgb"  # sRGB gamma encoded, [0.0, 1.0]


@dataclass
class ImageData:
    """Unified image container that flows through the entire pipeline.

    All image data is float32 RGB with shape (H, W, 3).
    - LINEAR: values in [0.0, potentially > 1.0] for HDR/RAW
    - SRGB: values in [0.0, 1.0], gamma-encoded
    """

    data: np.ndarray  # (H, W, 3) float32 RGB
    encoding: Encoding
    source_format: str  # "raw", "jpg", "heif", "png", "tiff"
    source_bit_depth: int  # 8, 10, 12, 14, 16
    original_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def is_linear(self) -> bool:
        return self.encoding == Encoding.LINEAR

    @property
    def is_srgb(self) -> bool:
        return self.encoding == Encoding.SRGB

    def to_linear(self) -> ImageData:
        """Convert to linear encoding. No-op if already linear."""
        if self.is_linear:
            return self
        return self.replace(data=srgb_to_linear(self.data), encoding=Encoding.LINEAR)

    def to_srgb(self) -> ImageData:
        """Convert to sRGB encoding, clipping to [0, 1]. No-op if already sRGB."""
        if self.is_srgb:
            return self
        return self.replace(
            data=np.clip(linear_to_srgb(self.data), 0.0, 1.0).astype(np.float32),
            encoding=Encoding.SRGB,
        )

    def replace(self, **kwargs: Any) -> ImageData:
        """Return a copy with specified fields replaced. Metadata is shallow-merged."""
        new_metadata = kwargs.pop("metadata", None)
        merged = {**self.metadata}
        if new_metadata is not None:
            merged.update(new_metadata)

        fields: dict[str, Any] = {
            "data": self.data,
            "encoding": self.encoding,
            "source_format": self.source_format,
            "source_bit_depth": self.source_bit_depth,
            "original_path": self.original_path,
            "metadata": merged,
        }
        fields.update(kwargs)
        return ImageData(**fields)
