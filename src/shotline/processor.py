"""Base processor interface and registry."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from enum import Enum
from typing import Any

from shotline.image import ImageData


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


class BaseProcessor(abc.ABC):
    @abc.abstractmethod
    def meta(self) -> ProcessorMeta: ...

    @abc.abstractmethod
    def process(self, image: ImageData, params: dict[str, Any] | None = None) -> ImageData: ...

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
