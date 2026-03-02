"""Auto color grading using Image-Adaptive 3D LUT."""

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
class ColorGradeProcessor(BaseProcessor):
    def meta(self) -> ProcessorMeta:
        return ProcessorMeta(
            name="color_grade",
            display_name="Auto Color Grade",
            description="Neural color grading (Image-Adaptive 3D LUT)",
            order=50,
            supported_inputs=["any"],
            requires_model=True,
            model_id="3dlut_color",
        )

    def status(self) -> ProcessorStatus:
        return ProcessorStatus.NEEDS_MODEL

    def process(self, image: ImageData, params: dict[str, Any] | None = None) -> ImageData:
        raise NotImplementedError("Color grade processor not yet implemented")
