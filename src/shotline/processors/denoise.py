"""AI denoising using NAFNet."""

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
class DenoiseProcessor(BaseProcessor):
    def meta(self) -> ProcessorMeta:
        return ProcessorMeta(
            name="denoise",
            display_name="AI Denoise",
            description="Neural network denoising (NAFNet)",
            order=20,
            supported_inputs=["any"],
            requires_model=True,
            model_id="nafnet_denoise",
        )

    def status(self) -> ProcessorStatus:
        return ProcessorStatus.NEEDS_MODEL

    def process(self, image: ImageData, params: dict[str, Any] | None = None) -> ImageData:
        raise NotImplementedError("Denoise processor not yet implemented")
