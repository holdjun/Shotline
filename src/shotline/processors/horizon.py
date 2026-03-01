"""Horizon/level correction using OpenCV."""

from __future__ import annotations

from typing import Any

import numpy as np

from shotline.processor import (
    BaseProcessor,
    ProcessorMeta,
    ProcessorStatus,
    ProcessResult,
    register_processor,
)


@register_processor
class HorizonProcessor(BaseProcessor):
    def meta(self) -> ProcessorMeta:
        return ProcessorMeta(
            name="horizon",
            display_name="Horizon Correction",
            description="Auto-detect and straighten horizon using Hough lines",
            order=30,
            supported_inputs=["any"],
        )

    def status(self) -> ProcessorStatus:
        try:
            import cv2  # noqa: F401

            return ProcessorStatus.AVAILABLE
        except ImportError:
            return ProcessorStatus.UNAVAILABLE

    def process(self, image: np.ndarray, params: dict[str, Any] | None = None) -> ProcessResult:
        # Stub: return image unchanged
        return ProcessResult(image=image, metadata={"stub": True})
