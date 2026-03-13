"""Exposure adjustment for JPG/HEIF: mild S-curve in sRGB space."""

from __future__ import annotations

from typing import Any

import numpy as np

from shotline.image import ImageData
from shotline.processor import BaseProcessor, ProcessorMeta, register_processor


def _mild_exposure_adjust(data: np.ndarray) -> np.ndarray:
    """Mild S-curve for already-sRGB images. Lifts shadows, tames highlights."""
    midpoint = 0.5
    strength = 0.1
    adjusted = data + strength * (data - midpoint) * (1.0 - data) * data * 4.0
    result: np.ndarray = np.clip(adjusted, 0.0, 1.0).astype(np.float32)
    return result


@register_processor
class ExposureAdjustProcessor(BaseProcessor):
    def meta(self) -> ProcessorMeta:
        return ProcessorMeta(
            name="exposure_adjust",
            display_name="Exposure Adjust",
            description="Mild exposure S-curve for JPG/HEIF images",
            order=11,
            supported_inputs=["jpg", "heif", "png", "tiff"],
        )

    def process(self, image: ImageData, params: dict[str, Any] | None = None) -> ImageData:  # noqa: ARG002
        adjusted = _mild_exposure_adjust(image.data)
        return image.replace(
            data=adjusted,
            metadata={"exposure_adjust": {"method": "mild_s_curve", "source": "srgb"}},
        )
