"""Pipeline orchestrator: chains processors on a single image."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from shotline.config import PipelineConfig, load_config
from shotline.image import ImageData
from shotline.io import load_image, save_image
from shotline.processor import BaseProcessor, ProcessorStatus, get_processor


class PipelineResult:
    def __init__(self) -> None:
        self.steps_run: list[dict[str, Any]] = []
        self.skipped: list[str] = []

    def to_dict(self) -> dict[str, Any]:
        return {"steps": self.steps_run, "skipped": self.skipped}


class Pipeline:
    def __init__(
        self,
        steps: list[str] | None = None,
        config: PipelineConfig | None = None,
    ):
        self.config = config or load_config()
        self.step_names = steps or self.config.default_steps
        self._processors: list[BaseProcessor] = []
        self._resolve()

    def _resolve(self) -> None:
        for name in self.step_names:
            self._processors.append(get_processor(name))

    def run(self, input_path: Path, output_path: Path) -> PipelineResult:
        result = PipelineResult()
        image: ImageData = load_image(input_path)

        for proc in self._processors:
            meta = proc.meta()

            if (
                "any" not in meta.supported_inputs
                and image.source_format not in meta.supported_inputs
            ):
                result.skipped.append(meta.name)
                continue

            if proc.status() in (ProcessorStatus.NEEDS_MODEL, ProcessorStatus.UNAVAILABLE):
                result.skipped.append(meta.name)
                continue

            params = self.config.get_processor_params(meta.name)
            t0 = time.perf_counter()
            image = proc.process(image, params)
            duration_ms = (time.perf_counter() - t0) * 1000

            result.steps_run.append(
                {
                    "name": meta.name,
                    "display_name": meta.display_name,
                    "duration_ms": round(duration_ms, 1),
                    "encoding": image.encoding.value,
                }
            )

        save_image(image, output_path, quality=self.config.output.quality)
        return result
