"""White balance correction."""

from __future__ import annotations

from typing import Any

import numpy as np

from shotline.processor import BaseProcessor, ProcessorMeta, ProcessResult, register_processor


@register_processor
class WhiteBalanceProcessor(BaseProcessor):
    def meta(self) -> ProcessorMeta:
        return ProcessorMeta(
            name="white_balance",
            display_name="White Balance",
            description="Auto white balance (gray world / AI model)",
            order=40,
            supported_inputs=["any"],
        )

    def process(self, image: np.ndarray, params: dict[str, Any] | None = None) -> ProcessResult:
        # Stub: return image unchanged
        return ProcessResult(image=image, metadata={"stub": True})
