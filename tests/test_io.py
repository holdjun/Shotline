"""Tests for image I/O."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from shotline.image import Encoding, ImageData
from shotline.io import (
    SUPPORTED_EXTENSIONS,
    _analyze_bayer,
    _compute_wb_compensation,
    detect_format,
    load_image,
    save_image,
)
from tests.conftest import TESTS_IMAGE_DIR, MockRawPy, make_bayer

# ── detect_format ──


def test_detect_format_jpg():
    assert detect_format(Path("photo.jpg")) == "jpg"
    assert detect_format(Path("photo.JPEG")) == "jpg"


def test_detect_format_raw():
    assert detect_format(Path("photo.cr2")) == "raw"
    assert detect_format(Path("photo.NEF")) == "raw"
    assert detect_format(Path("photo.dng")) == "raw"


def test_detect_format_heif():
    assert detect_format(Path("photo.heic")) == "heif"
    assert detect_format(Path("photo.HEIF")) == "heif"


def test_detect_format_unsupported():
    with pytest.raises(ValueError, match="Unsupported"):
        detect_format(Path("file.bmp"))


# ── load/save standard formats ──


def test_load_jpg(sample_jpg: Path):
    image = load_image(sample_jpg)
    assert isinstance(image, ImageData)
    assert image.source_format == "jpg"
    assert image.encoding == Encoding.SRGB
    assert image.data.dtype == np.float32
    assert image.data.shape == (100, 100, 3)
    assert 0.0 <= image.data.min() <= image.data.max() <= 1.0
    assert image.source_bit_depth == 8


def test_save_and_reload(sample_image: ImageData, tmp_path: Path):
    path = tmp_path / "out.jpg"
    save_image(sample_image, path)
    assert path.exists()

    image = load_image(path)
    assert image.data.shape[2] == 3
    assert image.data.dtype == np.float32


def test_save_png(sample_image: ImageData, tmp_path: Path):
    path = tmp_path / "out.png"
    save_image(sample_image, path)
    assert path.exists()

    image = load_image(path)
    assert image.source_format == "png"


def test_save_creates_parent_dirs(sample_image: ImageData, tmp_path: Path):
    path = tmp_path / "nested" / "dir" / "out.jpg"
    save_image(sample_image, path)
    assert path.exists()


def test_save_linear_auto_converts(sample_linear_image: ImageData, tmp_path: Path):
    """Saving a linear image auto-converts to sRGB."""
    path = tmp_path / "out.jpg"
    save_image(sample_linear_image, path)
    assert path.exists()

    image = load_image(path)
    assert image.encoding == Encoding.SRGB


def test_supported_extensions_complete():
    assert ".jpg" in SUPPORTED_EXTENSIONS
    assert ".cr2" in SUPPORTED_EXTENSIONS
    assert ".heic" in SUPPORTED_EXTENSIONS
    assert ".hif" in SUPPORTED_EXTENSIONS
    assert ".png" in SUPPORTED_EXTENSIONS


# ── _analyze_bayer tests ──


def test_analyze_bayer_clean_sensor():
    """Low saturation + low noise → Clip + Off."""
    bayer = make_bayer(saturation_ratio=0.0, noise_std=0.001)
    mock_raw = MockRawPy(bayer_data=bayer)
    analysis = _analyze_bayer(mock_raw)

    assert analysis["saturation_ratio"] < 0.005
    assert analysis["recommended_highlight"] == "Clip"
    assert analysis["recommended_fbdd"] == "Off"


def test_analyze_bayer_saturated():
    """High saturation → Blend."""
    bayer = make_bayer(saturation_ratio=0.02, noise_std=0.001)
    mock_raw = MockRawPy(bayer_data=bayer)
    analysis = _analyze_bayer(mock_raw)

    assert analysis["saturation_ratio"] >= 0.005
    assert analysis["recommended_highlight"] == "Blend"


def test_analyze_bayer_noisy():
    """High noise → Full FBDD."""
    bayer = make_bayer(saturation_ratio=0.0, noise_std=0.015)
    mock_raw = MockRawPy(bayer_data=bayer)
    analysis = _analyze_bayer(mock_raw)

    assert analysis["noise_std"] >= 0.008
    assert analysis["recommended_fbdd"] == "Full"


def test_analyze_bayer_medium_noise():
    """Medium noise → Light FBDD."""
    bayer = make_bayer(saturation_ratio=0.0, noise_std=0.005)
    mock_raw = MockRawPy(bayer_data=bayer)
    analysis = _analyze_bayer(mock_raw)

    assert 0.003 <= analysis["noise_std"] < 0.008
    assert analysis["recommended_fbdd"] == "Light"


# ── WB compensation tests ──


def test_wb_compensation_typical():
    """Typical Sony WB gives ~2.1x compensation."""
    wb = [2.1, 1.0, 1.5, 1.0]
    comp = _compute_wb_compensation(wb)
    assert comp == pytest.approx(2.1, rel=0.01)


def test_wb_compensation_equal():
    """Equal WB channels → 1.0 compensation."""
    assert _compute_wb_compensation([1.0, 1.0, 1.0, 1.0]) == pytest.approx(1.0)


def test_wb_compensation_zeros():
    """All-zero WB → 1.0 (safe fallback)."""
    assert _compute_wb_compensation([0.0, 0.0, 0.0, 0.0]) == 1.0


# ── _load_raw mock tests ──


def test_load_raw_default_params():
    """Default params produce LINEAR ImageData with bayer_analysis."""
    mock_raw = MockRawPy()
    with patch("rawpy.imread", return_value=mock_raw):
        image = load_image(Path("test.arw"))

    assert image.encoding == Encoding.LINEAR
    assert image.source_format == "raw"
    assert image.data.dtype == np.float32
    assert image.data.shape == (100, 100, 3)
    assert "raw_loader" in image.metadata

    meta = image.metadata["raw_loader"]
    assert meta["demosaic_algorithm"] == "DHT"
    assert meta["used_camera_wb"] is True
    assert meta["used_auto_wb"] is False
    assert "bayer_analysis" in meta
    assert "saturation_ratio" in meta["bayer_analysis"]


def test_load_raw_adaptive_clip():
    """Clean sensor auto-selects Clip mode (no WB compensation)."""
    bayer = make_bayer(saturation_ratio=0.0, noise_std=0.001)
    mock_raw = MockRawPy(bayer_data=bayer)
    with patch("rawpy.imread", return_value=mock_raw):
        image = load_image(Path("test.arw"))

    meta = image.metadata["raw_loader"]
    assert meta["highlight_mode"] == "Clip"
    assert meta["wb_compensation"] == 1.0


def test_load_raw_adaptive_blend_with_compensation():
    """Saturated sensor auto-selects Blend with WB compensation."""
    bayer = make_bayer(saturation_ratio=0.02, noise_std=0.001)
    mock_raw = MockRawPy(bayer_data=bayer)
    with patch("rawpy.imread", return_value=mock_raw):
        image = load_image(Path("test.arw"))

    meta = image.metadata["raw_loader"]
    assert meta["highlight_mode"] == "Blend"
    assert meta["wb_compensation"] > 1.0
    # Data should have values > 1.0 from WB compensation
    assert image.data.max() > 1.0


def test_load_raw_user_override_highlight():
    """User explicit highlight_mode overrides adaptive choice."""
    bayer = make_bayer(saturation_ratio=0.0, noise_std=0.001)
    mock_raw = MockRawPy(bayer_data=bayer)
    with patch("rawpy.imread", return_value=mock_raw):
        # Bayer says Clip, but user forces Blend
        image = load_image(Path("test.arw"), raw_params={"highlight_mode": "Blend"})

    meta = image.metadata["raw_loader"]
    assert meta["highlight_mode"] == "Blend"
    assert meta["wb_compensation"] > 1.0


def test_load_raw_user_override_fbdd():
    """User explicit fbdd overrides adaptive choice."""
    bayer = make_bayer(saturation_ratio=0.0, noise_std=0.001)
    mock_raw = MockRawPy(bayer_data=bayer)
    with patch("rawpy.imread", return_value=mock_raw):
        # Bayer says Off, but user forces Full
        image = load_image(Path("test.arw"), raw_params={"fbdd_noise_reduction": "Full"})

    meta = image.metadata["raw_loader"]
    assert meta["fbdd_noise_reduction"] == "Full"


def test_load_raw_with_ev():
    """ev=1.0 translates to exp_shift=2.0 in rawpy."""
    mock_raw = MockRawPy()
    with patch("rawpy.imread", return_value=mock_raw):
        image = load_image(Path("test.arw"), raw_params={"ev": 1.0})

    assert image.metadata["raw_loader"]["exp_shift_applied"] == 2.0
    assert mock_raw._last_pp_kwargs["exp_shift"] == 2.0


def test_load_raw_wb_fallback():
    """camera_wb all zeros triggers auto_wb fallback."""
    mock_raw = MockRawPy(camera_wb=[0.0, 0.0, 0.0, 0.0])
    with patch("rawpy.imread", return_value=mock_raw):
        image = load_image(Path("test.arw"))

    meta = image.metadata["raw_loader"]
    assert meta["used_camera_wb"] is False
    assert meta["used_auto_wb"] is True
    assert mock_raw._last_pp_kwargs["use_camera_wb"] is False
    assert mock_raw._last_pp_kwargs["use_auto_wb"] is True


def test_load_raw_metadata_fields():
    """All expected metadata fields are present."""
    mock_raw = MockRawPy()
    with patch("rawpy.imread", return_value=mock_raw):
        image = load_image(Path("test.arw"))

    meta = image.metadata["raw_loader"]
    expected_keys = {
        "camera_whitebalance",
        "daylight_whitebalance",
        "black_level_per_channel",
        "white_level",
        "color_desc",
        "num_colors",
        "sizes",
        "demosaic_algorithm",
        "highlight_mode",
        "fbdd_noise_reduction",
        "exp_shift_applied",
        "exp_preserve_highlights",
        "used_camera_wb",
        "used_auto_wb",
        "wb_compensation",
        "bayer_analysis",
    }
    assert set(meta.keys()) == expected_keys
    assert meta["white_level"] == 16383
    assert meta["color_desc"] == "RGBG"
    assert meta["sizes"] == {"width": 100, "height": 100}


def test_load_raw_custom_params():
    """Custom demosaic/highlight/fbdd params are passed to rawpy."""
    mock_raw = MockRawPy()
    with patch("rawpy.imread", return_value=mock_raw):
        load_image(
            Path("test.arw"),
            raw_params={
                "demosaic_algorithm": "AHD",
                "highlight_mode": "Clip",
                "fbdd_noise_reduction": "Full",
                "median_filter_passes": 3,
            },
        )

    pp = mock_raw._last_pp_kwargs
    assert pp["median_filter_passes"] == 3


# ── Real image integration tests ──


def _skip_unless_exists(path: Path) -> pytest.MarkDecorator:
    return pytest.mark.skipif(not path.exists(), reason=f"{path.name} not found in tests/image/")


_ARW_1 = TESTS_IMAGE_DIR / "DSC00078.ARW"
_ARW_2 = TESTS_IMAGE_DIR / "DSC04188.ARW"
_JPG = TESTS_IMAGE_DIR / "2942527655.jpg"
_HIF = TESTS_IMAGE_DIR / "DSC04188.HIF"


@_skip_unless_exists(_ARW_1)
@pytest.mark.slow
def test_load_real_arw():
    """Load a real Sony ARW file."""
    image = load_image(_ARW_1)
    assert image.encoding == Encoding.LINEAR
    assert image.source_format == "raw"
    assert image.data.dtype == np.float32
    assert image.data.ndim == 3
    assert image.data.shape[2] == 3

    meta = image.metadata["raw_loader"]
    assert meta["demosaic_algorithm"] == "DHT"
    assert meta["used_camera_wb"] is True
    assert "bayer_analysis" in meta


@_skip_unless_exists(_ARW_2)
@pytest.mark.slow
def test_load_real_arw_with_ev():
    """Loading with ev=1.0 produces brighter output than ev=0."""
    img_base = load_image(_ARW_2)
    img_bright = load_image(_ARW_2, raw_params={"ev": 1.0})

    assert img_bright.metadata["raw_loader"]["exp_shift_applied"] == 2.0
    assert img_bright.data.mean() > img_base.data.mean()


@_skip_unless_exists(_ARW_2)
@pytest.mark.slow
def test_load_real_arw_adaptive_highlight():
    """DSC04188 has overexposed highlights → adaptive selects Blend."""
    image = load_image(_ARW_2)
    meta = image.metadata["raw_loader"]
    analysis = meta["bayer_analysis"]
    # DSC04188 is known to have ~1.8% saturation
    assert analysis["saturation_ratio"] > 0.005
    assert meta["highlight_mode"] == "Blend"
    assert meta["wb_compensation"] > 1.0


@_skip_unless_exists(_JPG)
@pytest.mark.slow
def test_load_real_jpg():
    """Load a real JPEG file."""
    image = load_image(_JPG)
    assert image.encoding == Encoding.SRGB
    assert image.source_format == "jpg"
    assert image.data.dtype == np.float32
    assert image.data.ndim == 3


@_skip_unless_exists(_HIF)
@pytest.mark.slow
def test_load_real_heif():
    """Load a real HEIF file."""
    image = load_image(_HIF)
    assert image.encoding == Encoding.SRGB
    assert image.source_format == "heif"
    assert image.data.dtype == np.float32
    assert image.data.ndim == 3


@_skip_unless_exists(_ARW_1)
@pytest.mark.slow
def test_real_arw_pipeline_roundtrip(tmp_path: Path):
    """Full pipeline roundtrip: ARW → processors → JPG."""
    import shotline.processors  # noqa: F401
    from shotline.config import PipelineConfig
    from shotline.pipeline import Pipeline

    out = tmp_path / "roundtrip.jpg"
    config = PipelineConfig(default_steps=["raw_develop", "exposure_adjust", "white_balance"])
    pipeline = Pipeline(config=config)
    result = pipeline.run(_ARW_1, out)

    assert out.exists()
    assert any(s["name"] == "raw_develop" for s in result.steps_run)
    # exposure_adjust skipped for RAW (only handles jpg/heif/png/tiff)
    assert "exposure_adjust" in result.skipped

    from PIL import Image

    img = Image.open(out)
    assert img.size[0] > 0
    assert img.size[1] > 0


@_skip_unless_exists(_ARW_2)
@pytest.mark.slow
def test_real_arw_adaptive_pipeline(tmp_path: Path):
    """Adaptive pipeline on high-DR image produces reasonable brightness."""
    import shotline.processors  # noqa: F401
    from shotline.config import PipelineConfig
    from shotline.pipeline import Pipeline

    out = tmp_path / "adaptive.jpg"
    config = PipelineConfig(default_steps=["raw_develop", "exposure_adjust"])
    pipeline = Pipeline(config=config)
    pipeline.run(_ARW_2, out)

    assert out.exists()
    # Reload and check brightness is reasonable (not too dark, not blown)
    reloaded = load_image(out)
    mean_brightness = float(reloaded.data.mean())
    assert 0.15 <= mean_brightness <= 0.65
