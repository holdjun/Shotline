"""Tone mapping: compress dynamic range and unify encoding to sRGB."""

from __future__ import annotations

from typing import Any

import numpy as np

from shotline.image import Encoding, ImageData, linear_to_srgb
from shotline.processor import BaseProcessor, ProcessorMeta, register_processor


def _filmic_curve(x: np.ndarray) -> np.ndarray:
    """Extended Reinhard tone mapping with filmic shoulder.

    Maps linear HDR [0, inf) → [0, 1) with a natural roll-off.
    """
    # Attempt to preserve midtones while compressing highlights
    white_point = 4.0  # values above this are deep in the shoulder
    numerator = x * (1.0 + x / (white_point * white_point))
    denominator = 1.0 + x
    return (numerator / denominator).astype(np.float32)


def _mild_exposure_adjust(data: np.ndarray) -> np.ndarray:
    """Mild S-curve for already-sRGB images. Lifts shadows, tames highlights."""
    midpoint = 0.5
    strength = 0.1
    adjusted = data + strength * (data - midpoint) * (1.0 - data) * data * 4.0
    return np.clip(adjusted, 0.0, 1.0).astype(np.float32)


@register_processor
class ToneMapProcessor(BaseProcessor):
    def meta(self) -> ProcessorMeta:
        return ProcessorMeta(
            name="tone_map",
            display_name="Tone Map",
            description="Compress dynamic range (filmic for RAW, mild adjust for JPG/HEIF)",
            order=15,
            supported_inputs=["any"],
        )

    def process(self, image: ImageData, params: dict[str, Any] | None = None) -> ImageData:
        if image.is_linear:
            # RAW path: filmic tone map in linear space, then convert to sRGB
            mapped = _filmic_curve(image.data)
            srgb = np.clip(linear_to_srgb(mapped), 0.0, 1.0).astype(np.float32)
            return image.replace(
                data=srgb,
                encoding=Encoding.SRGB,
                metadata={"tone_map": {"method": "filmic", "source": "linear"}},
            )
        else:
            # JPG/HEIF path: mild exposure adjustment in sRGB
            adjusted = _mild_exposure_adjust(image.data)
            return image.replace(
                data=adjusted,
                metadata={"tone_map": {"method": "mild_exposure", "source": "srgb"}},
            )
