"""Base processor interface and registry."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class ProcessorStatus(Enum):
    AVAILABLE = "available"
    NEEDS_MODEL = "needs_model"
    UNAVAILABLE = "unavailable"


@dataclass
class ProcessorMeta:
    name: str
    display_name: str
    description: str
    order: int  # pipeline position: 10, 20, 30...
    supported_inputs: list[str]  # ["raw", "jpg", "heif"] or ["any"]
    requires_model: bool = False
    model_id: str | None = None


@dataclass
class ProcessResult:
    image: np.ndarray  # (H, W, C) float32 [0, 1]
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseProcessor(abc.ABC):
    @abc.abstractmethod
    def meta(self) -> ProcessorMeta: ...

    @abc.abstractmethod
    def process(self, image: np.ndarray, params: dict[str, Any] | None = None) -> ProcessResult: ...

    def status(self) -> ProcessorStatus:
        return ProcessorStatus.AVAILABLE


# ── Registry ──

_REGISTRY: dict[str, type[BaseProcessor]] = {}


def register_processor(cls: type[BaseProcessor]) -> type[BaseProcessor]:
    instance = cls()
    _REGISTRY[instance.meta().name] = cls
    return cls


def get_processor(name: str) -> BaseProcessor:
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise ValueError(f"Unknown processor: {name}. Available: {available}")
    return _REGISTRY[name]()


def list_processors() -> list[ProcessorMeta]:
    metas = [cls().meta() for cls in _REGISTRY.values()]
    return sorted(metas, key=lambda m: m.order)
