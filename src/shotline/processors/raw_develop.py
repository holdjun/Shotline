"""RAW development using rawpy."""

from __future__ import annotations

from typing import Any

import numpy as np

from shotline.processor import BaseProcessor, ProcessorMeta, ProcessResult, register_processor


@register_processor
class RawDevelopProcessor(BaseProcessor):
    def meta(self) -> ProcessorMeta:
        return ProcessorMeta(
            name="raw_develop",
            display_name="RAW Development",
            description="Develop RAW files: demosaic, highlight/shadow recovery",
            order=10,
            supported_inputs=["raw"],
        )

    def process(self, image: np.ndarray, params: dict[str, Any] | None = None) -> ProcessResult:
        # Stub: image already loaded via rawpy in io.py
        return ProcessResult(image=image, metadata={"stub": True})
