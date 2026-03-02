"""Tests for processor registry and interface."""

from __future__ import annotations

import numpy as np

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


def test_raw_develop_passthrough(sample_image: ImageData):
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


# ── raw_develop detailed tests ──


def test_raw_develop_ev_zero(sample_linear_image: ImageData):
    """ev=0 does not change the data."""
    proc = get_processor("raw_develop")
    result = proc.process(sample_linear_image, {"ev": 0.0})
    np.testing.assert_array_equal(result.data, sample_linear_image.data)


def test_raw_develop_ev_fallback(sample_linear_image: ImageData):
    """Without raw_loader metadata, EV fallback applies 2^ev multiplier."""
    proc = get_processor("raw_develop")
    result = proc.process(sample_linear_image, {"ev": 1.0})
    expected = np.maximum(sample_linear_image.data * 2.0, 0.0).astype(np.float32)
    np.testing.assert_allclose(result.data, expected, rtol=1e-5)


def test_raw_develop_ev_skip_when_loaded(sample_linear_image_with_metadata: ImageData):
    """When exp_shift was applied at load, raw_develop skips EV multiplication."""
    proc = get_processor("raw_develop")
    result = proc.process(sample_linear_image_with_metadata, {"ev": 1.0})
    # exp_shift_applied == 2.0 in fixture → EV not applied again
    np.testing.assert_array_equal(result.data, sample_linear_image_with_metadata.data)
    assert result.metadata["raw_develop"]["exp_shift_applied_at_load"] is True


def test_raw_develop_brightness(sample_linear_image: ImageData):
    """bright parameter scales data linearly."""
    proc = get_processor("raw_develop")
    result = proc.process(sample_linear_image, {"bright": 1.5})
    expected = np.maximum(sample_linear_image.data * 1.5, 0.0).astype(np.float32)
    np.testing.assert_allclose(result.data, expected, rtol=1e-5)


def test_raw_develop_no_negative_output(sample_linear_image: ImageData):
    """Output has no negative values even with negative EV."""
    proc = get_processor("raw_develop")
    result = proc.process(sample_linear_image, {"ev": -2.0})
    assert result.data.min() >= 0.0


def test_raw_develop_metadata(sample_linear_image: ImageData):
    """raw_develop records its parameters in metadata."""
    proc = get_processor("raw_develop")
    result = proc.process(sample_linear_image, {"ev": 0.5, "bright": 1.2})
    meta = result.metadata["raw_develop"]
    assert meta["ev"] == 0.5
    assert meta["bright"] == 1.2
    assert meta["exp_shift_applied_at_load"] is False
