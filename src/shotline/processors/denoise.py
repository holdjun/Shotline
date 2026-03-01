"""AI denoising using NAFNet."""

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
        # TODO: check if model is downloaded
        return ProcessorStatus.NEEDS_MODEL

    def process(self, image: np.ndarray, params: dict[str, Any] | None = None) -> ProcessResult:
        raise NotImplementedError("Denoise processor not yet implemented")
