"""Configuration loading from TOML."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATHS = [
    Path("shotline.toml"),
    Path.home() / ".config" / "shotline" / "config.toml",
]

DEFAULT_STEPS = [
    "raw_develop",
    "denoise",
    "horizon",
    "white_balance",
    "color_grade",
    "auto_crop",
]


class OutputConfig(BaseModel):
    format: str = "jpg"
    quality: int = 95
    suffix: str = "_processed"


class ModelConfig(BaseModel):
    cache_dir: Path = Field(default=Path.home() / ".cache" / "shotline" / "models")


class PipelineConfig(BaseModel):
    default_steps: list[str] = Field(default_factory=lambda: list(DEFAULT_STEPS))
    output: OutputConfig = Field(default_factory=OutputConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    processor_params: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def get_processor_params(self, name: str) -> dict[str, Any]:
        return self.processor_params.get(name, {})


def load_config(path: Path | None = None) -> PipelineConfig:
    if path and path.exists():
        return _parse_toml(path)
    for candidate in DEFAULT_CONFIG_PATHS:
        if candidate.exists():
            return _parse_toml(candidate)
    return PipelineConfig()


def _parse_toml(path: Path) -> PipelineConfig:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return PipelineConfig.model_validate(data)
