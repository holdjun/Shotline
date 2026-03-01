"""White balance correction."""

from __future__ import annotations

from typing import Any

from shotline.image import ImageData
from shotline.processor import BaseProcessor, ProcessorMeta, register_processor


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

    def process(self, image: ImageData, params: dict[str, Any] | None = None) -> ImageData:
        return image.replace(metadata={"white_balance": {"stub": True}})
