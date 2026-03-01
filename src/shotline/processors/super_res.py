"""Super resolution using Real-ESRGAN."""

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
class SuperResProcessor(BaseProcessor):
    def meta(self) -> ProcessorMeta:
        return ProcessorMeta(
            name="super_res",
            display_name="Super Resolution",
            description="AI upscaling (Real-ESRGAN x4)",
            order=70,
            supported_inputs=["any"],
            requires_model=True,
            model_id="real_esrgan",
        )

    def status(self) -> ProcessorStatus:
        return ProcessorStatus.NEEDS_MODEL

    def process(self, image: np.ndarray, params: dict[str, Any] | None = None) -> ProcessResult:
        raise NotImplementedError("Super resolution processor not yet implemented")
