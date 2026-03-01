"""Tests for processor registry and interface."""

from __future__ import annotations

import shotline.processors  # noqa: F401
from shotline.image import ImageData
from shotline.processor import (
    BaseProcessor,
    ProcessorMeta,
    ProcessorStatus,
    get_processor,
    list_processors,
)


def test_list_processors_returns_all():
    metas = list_processors()
    names = [m.name for m in metas]
    assert "raw_develop" in names
    assert "tone_map" in names
    assert "denoise" in names
    assert "horizon" in names
    assert "white_balance" in names
    assert "color_grade" in names
    assert "auto_crop" in names
    assert "super_res" in names
    assert "content_remove" in names
    assert len(metas) == 9


def test_list_processors_sorted_by_order():
    metas = list_processors()
    orders = [m.order for m in metas]
    assert orders == sorted(orders)


def test_get_processor_returns_instance():
    proc = get_processor("raw_develop")
    assert isinstance(proc, BaseProcessor)
    assert proc.meta().name == "raw_develop"


def test_get_unknown_processor_raises():
    import pytest

    with pytest.raises(ValueError, match="Unknown processor"):
        get_processor("nonexistent")


def test_raw_develop_stub(sample_image: ImageData):
    proc = get_processor("raw_develop")
    # raw_develop expects "raw" source_format, but test still exercises the code path
    result = proc.process(sample_image)
    assert isinstance(result, ImageData)
    assert result.data.shape == sample_image.data.shape


def test_white_balance_stub(sample_image: ImageData):
    proc = get_processor("white_balance")
    result = proc.process(sample_image)
    assert result.data.shape == sample_image.data.shape


def test_tone_map_srgb(sample_image: ImageData):
    proc = get_processor("tone_map")
    result = proc.process(sample_image)
    assert result.is_srgb
    assert result.data.shape == sample_image.data.shape


def test_tone_map_linear(sample_linear_image: ImageData):
    proc = get_processor("tone_map")
    result = proc.process(sample_linear_image)
    assert result.is_srgb
    assert result.data.min() >= 0.0
    assert result.data.max() <= 1.0


def test_needs_model_processors():
    for name in ["denoise", "color_grade", "auto_crop", "super_res"]:
        proc = get_processor(name)
        assert proc.status() == ProcessorStatus.NEEDS_MODEL


def test_content_remove_unavailable():
    proc = get_processor("content_remove")
    assert proc.status() == ProcessorStatus.UNAVAILABLE


def test_processor_meta_fields():
    proc = get_processor("raw_develop")
    meta = proc.meta()
    assert isinstance(meta, ProcessorMeta)
    assert isinstance(meta.name, str)
    assert isinstance(meta.display_name, str)
    assert isinstance(meta.description, str)
    assert isinstance(meta.order, int)
    assert isinstance(meta.supported_inputs, list)


def test_all_processors_have_unique_names():
    metas = list_processors()
    names = [m.name for m in metas]
    assert len(names) == len(set(names))


def test_all_processors_have_unique_orders():
    metas = list_processors()
    orders = [m.order for m in metas]
    assert len(orders) == len(set(orders))
