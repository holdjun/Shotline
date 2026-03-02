"""RAW development: exposure fallback and brightness in linear light."""

from __future__ import annotations

from typing import Any

import numpy as np

from shotline.image import ImageData
from shotline.processor import BaseProcessor, ProcessorMeta, register_processor


@register_processor
class RawDevelopProcessor(BaseProcessor):
    def meta(self) -> ProcessorMeta:
        return ProcessorMeta(
            name="raw_develop",
            display_name="RAW Development",
            description="Develop RAW files: exposure fallback and brightness in linear light",
            order=10,
            supported_inputs=["raw"],
        )

    def process(self, image: ImageData, params: dict[str, Any] | None = None) -> ImageData:
        params = params or {}
        ev = float(params.get("ev", 0.0))
        bright = float(params.get("bright", 1.0))

        data = image.data
        loader_meta = image.metadata.get("raw_loader", {})

        # Fallback EV: only apply if _load_raw did not use exp_shift
        if ev != 0.0 and loader_meta.get("exp_shift_applied", 1.0) == 1.0:
            data = data * (2.0**ev)

        # Brightness scaling (independent linear multiplier)
        if bright != 1.0:
            data = data * bright

        # Ensure non-negative
        if ev != 0.0 or bright != 1.0:
            data = np.maximum(data, 0.0).astype(np.float32)

        return image.replace(
            data=data,
            metadata={
                "raw_develop": {
                    "ev": ev,
                    "bright": bright,
                    "exp_shift_applied_at_load": loader_meta.get("exp_shift_applied", 1.0) != 1.0,
                }
            },
        )
