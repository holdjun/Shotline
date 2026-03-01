"""Smart auto-crop using subject detection."""

from __future__ import annotations

from typing import Any

from shotline.image import ImageData
from shotline.processor import (
    BaseProcessor,
    ProcessorMeta,
    ProcessorStatus,
    register_processor,
)


@register_processor
class AutoCropProcessor(BaseProcessor):
    def meta(self) -> ProcessorMeta:
        return ProcessorMeta(
            name="auto_crop",
            display_name="Auto Crop",
            description="Smart cropping with subject detection and rule of thirds",
            order=60,
            supported_inputs=["any"],
            requires_model=True,
            model_id="yolov8_crop",
        )

    def status(self) -> ProcessorStatus:
        return ProcessorStatus.NEEDS_MODEL

    def process(self, image: ImageData, params: dict[str, Any] | None = None) -> ImageData:
        raise NotImplementedError("Auto crop processor not yet implemented")
