"""RAW development: exposure compensation in linear light."""

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
            description="Develop RAW files: exposure compensation in linear light",
            order=10,
            supported_inputs=["raw"],
        )

    def process(self, image: ImageData, params: dict[str, Any] | None = None) -> ImageData:
        params = params or {}
        ev = params.get("ev", 0.0)
        data = image.data
        if ev != 0.0:
            data = data * (2.0 ** ev)
            data = np.clip(data, 0.0, None).astype(np.float32)
        return image.replace(data=data, metadata={"raw_develop": {"ev": ev}})
