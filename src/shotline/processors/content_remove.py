"""Content-aware removal using LaMa/IOPaint (future)."""

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
class ContentRemoveProcessor(BaseProcessor):
    def meta(self) -> ProcessorMeta:
        return ProcessorMeta(
            name="content_remove",
            display_name="Content-Aware Remove",
            description="AI inpainting to remove unwanted objects (interactive, future)",
            order=80,
            supported_inputs=["any"],
            requires_model=True,
            model_id=None,
        )

    def status(self) -> ProcessorStatus:
        return ProcessorStatus.UNAVAILABLE

    def process(self, image: ImageData, params: dict[str, Any] | None = None) -> ImageData:
        raise NotImplementedError("Content removal is not yet implemented")
