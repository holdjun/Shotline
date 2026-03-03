"""Tests for processor registry and interface."""

from __future__ import annotations

import numpy as np

import shotline.processors  # noqa: F401
from shotline.image import Encoding, ImageData
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
    assert "exposure_adjust" in names
    assert "denoise" in names
    assert "horizon" in names
    assert "white_balance" in names
    assert "color_grade" in names
    assert "auto_crop" in names
    assert "super_res" in names
    assert "content_remove" in names
    assert "lens_correct" in names
    assert len(metas) == 10


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
    result = proc.process(sample_image)
    assert isinstance(result, ImageData)
    assert result.data.shape == sample_image.data.shape


def test_white_balance_stub(sample_image: ImageData):
    proc = get_processor("white_balance")
    result = proc.process(sample_image)
    assert result.data.shape == sample_image.data.shape


def test_exposure_adjust_srgb(sample_image: ImageData):
    proc = get_processor("exposure_adjust")
    result = proc.process(sample_image)
    assert result.is_srgb
    assert result.data.shape == sample_image.data.shape
    meta = result.metadata["exposure_adjust"]
    assert meta["method"] == "mild_s_curve"


def test_raw_develop_linear_outputs_srgb(sample_linear_image: ImageData):
    proc = get_processor("raw_develop")
    result = proc.process(sample_linear_image)
    assert result.is_srgb
    assert result.data.min() >= 0.0
    assert result.data.max() <= 1.0
    meta = result.metadata["raw_develop"]
    assert meta["tone_map"]["method"] == "hable_filmic"


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
    """ev=0, auto_expose=False: only Hable filmic + sRGB applied."""
    proc = get_processor("raw_develop")
    result = proc.process(sample_linear_image, {"ev": 0.0, "auto_expose": False})
    assert result.encoding == Encoding.SRGB
    assert result.data.min() >= 0.0
    assert result.data.max() <= 1.0


def test_raw_develop_ev_fallback(sample_linear_image: ImageData):
    """Without raw_loader metadata, EV applies 2^ev in linear before tone map."""
    proc = get_processor("raw_develop")
    result_ev0 = proc.process(sample_linear_image, {"ev": 0.0, "auto_expose": False})
    result_ev1 = proc.process(sample_linear_image, {"ev": 1.0, "auto_expose": False})
    # +1 EV should produce brighter sRGB output
    assert result_ev1.data.mean() > result_ev0.data.mean()


def test_raw_develop_ev_skip_when_loaded(sample_linear_image_with_metadata: ImageData):
    """When exp_shift was applied at load, raw_develop skips EV multiplication."""
    proc = get_processor("raw_develop")
    result = proc.process(sample_linear_image_with_metadata, {"ev": 1.0, "auto_expose": False})
    assert result.metadata["raw_develop"]["exp_shift_applied_at_load"] is True


def test_raw_develop_brightness(sample_linear_image: ImageData):
    """bright > 1.0 produces brighter sRGB output."""
    proc = get_processor("raw_develop")
    result_b1 = proc.process(sample_linear_image, {"bright": 1.0, "auto_expose": False})
    result_b2 = proc.process(sample_linear_image, {"bright": 2.0, "auto_expose": False})
    assert result_b2.data.mean() > result_b1.data.mean()


def test_raw_develop_no_negative_output(sample_linear_image: ImageData):
    """Output has no negative values even with negative EV."""
    proc = get_processor("raw_develop")
    result = proc.process(sample_linear_image, {"ev": -2.0, "auto_expose": False})
    assert result.data.min() >= 0.0


def test_raw_develop_metadata(sample_linear_image: ImageData):
    """raw_develop records its parameters in metadata."""
    proc = get_processor("raw_develop")
    result = proc.process(sample_linear_image, {"ev": 0.5, "bright": 1.2, "auto_expose": False})
    meta = result.metadata["raw_develop"]
    assert meta["ev"] == 0.5
    assert meta["bright"] == 1.2
    assert meta["exp_shift_applied_at_load"] is False
    assert meta["auto_expose"] is False
    assert meta["auto_ev"] == 0.0
    assert meta["tone_map"]["method"] == "hable_filmic"
    assert meta["tone_map"]["white_point"] == 11.2


# ── Auto-exposure tests ──


def _make_uniform_linear_image(value: float) -> ImageData:
    """Create a uniform-brightness linear image for auto-exposure testing."""
    data = np.full((100, 100, 3), value, dtype=np.float32)
    return ImageData(
        data=data,
        encoding=Encoding.LINEAR,
        source_format="raw",
        source_bit_depth=16,
    )


def test_auto_ev_dark_image():
    """Dark image → auto_ev > 0 (brightens)."""
    image = _make_uniform_linear_image(0.02)
    proc = get_processor("raw_develop")
    result = proc.process(image, {"auto_expose": True})
    meta = result.metadata["raw_develop"]
    assert meta["auto_ev"] > 0.0


def test_auto_ev_bright_image():
    """Bright image → auto_ev < 0 (darkens)."""
    image = _make_uniform_linear_image(0.8)
    proc = get_processor("raw_develop")
    result = proc.process(image, {"auto_expose": True})
    meta = result.metadata["raw_develop"]
    assert meta["auto_ev"] < 0.0


def test_auto_ev_midgray():
    """18% gray → auto_ev ≈ 0."""
    image = _make_uniform_linear_image(0.18)
    proc = get_processor("raw_develop")
    result = proc.process(image, {"auto_expose": True})
    meta = result.metadata["raw_develop"]
    assert abs(meta["auto_ev"]) < 0.5


def test_auto_expose_disabled():
    """auto_expose=False → auto_ev=0."""
    image = _make_uniform_linear_image(0.02)
    proc = get_processor("raw_develop")
    result = proc.process(image, {"auto_expose": False})
    meta = result.metadata["raw_develop"]
    assert meta["auto_ev"] == 0.0


def test_manual_ev_stacks_on_auto():
    """Manual EV stacks on top of auto-exposure."""
    image = _make_uniform_linear_image(0.18)
    proc = get_processor("raw_develop")

    result_auto_only = proc.process(image, {"auto_expose": True, "ev": 0.0})
    result_auto_plus_ev = proc.process(image, {"auto_expose": True, "ev": 1.0})

    # +1 EV should produce brighter output
    assert result_auto_plus_ev.data.mean() > result_auto_only.data.mean()


def test_auto_ev_stats_recorded():
    """Auto-exposure records scene_key and percentiles in metadata."""
    image = _make_uniform_linear_image(0.1)
    proc = get_processor("raw_develop")
    result = proc.process(image, {"auto_expose": True})
    stats = result.metadata["raw_develop"]["auto_ev_stats"]
    assert "scene_key" in stats
    assert "p1" in stats
    assert "p99" in stats
    assert stats["scene_key"] > 0


# ── Hable filmic tests ──


def test_hable_zero_input():
    """Input 0 → output ≈ 0 (toe starts near zero)."""
    from shotline.processors.raw_develop import _hable_filmic

    data = np.zeros((10, 10, 3), dtype=np.float32)
    result = _hable_filmic(data)
    assert result.max() < 0.05


def test_hable_hdr_compression():
    """HDR input (3.0) is compressed to < 1.0."""
    from shotline.processors.raw_develop import _hable_filmic

    data = np.full((10, 10, 3), 3.0, dtype=np.float32)
    result = _hable_filmic(data)
    assert result.max() < 1.0
    assert result.min() > 0.5


def test_hable_output_range():
    """Output is always in [0, 1]."""
    from shotline.processors.raw_develop import _hable_filmic

    rng = np.random.default_rng(42)
    data = rng.random((50, 50, 3), dtype=np.float32) * 5.0
    result = _hable_filmic(data)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_hable_white_point_effect():
    """Larger white point → dimmer output (more headroom)."""
    from shotline.processors.raw_develop import _hable_filmic

    data = np.full((10, 10, 3), 2.0, dtype=np.float32)
    result_small_w = _hable_filmic(data, white_point=4.0)
    result_large_w = _hable_filmic(data, white_point=20.0)
    assert result_small_w.mean() > result_large_w.mean()


def test_hable_monotonic():
    """Hable filmic is monotonically increasing."""
    from shotline.processors.raw_develop import _hable_filmic

    values = np.linspace(0, 10, 100).reshape(1, 100, 1).repeat(3, axis=2).astype(np.float32)
    result = _hable_filmic(values)
    diffs = np.diff(result[0, :, 0])
    assert np.all(diffs >= 0)


def test_raw_develop_custom_white_point(sample_linear_image: ImageData):
    """Custom white_point via params."""
    proc = get_processor("raw_develop")
    result = proc.process(sample_linear_image, {"white_point": 6.0, "auto_expose": False})
    meta = result.metadata["raw_develop"]["tone_map"]
    assert meta["white_point"] == 6.0
