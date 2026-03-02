from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from shotline.image import Encoding, ImageData


@pytest.fixture
def sample_image() -> ImageData:
    """100x100 sRGB ImageData test image."""
    rng = np.random.default_rng(42)
    data = rng.random((100, 100, 3), dtype=np.float32)
    return ImageData(
        data=data,
        encoding=Encoding.SRGB,
        source_format="jpg",
        source_bit_depth=8,
    )


@pytest.fixture
def sample_linear_image() -> ImageData:
    """100x100 linear ImageData test image (simulates RAW)."""
    rng = np.random.default_rng(42)
    data = rng.random((100, 100, 3), dtype=np.float32) * 1.5  # allow > 1.0
    return ImageData(
        data=data,
        encoding=Encoding.LINEAR,
        source_format="raw",
        source_bit_depth=16,
    )


@pytest.fixture
def sample_linear_image_with_metadata() -> ImageData:
    """100x100 linear ImageData with raw_loader metadata (simulates _load_raw output)."""
    rng = np.random.default_rng(42)
    data = rng.random((100, 100, 3), dtype=np.float32) * 1.5
    return ImageData(
        data=data,
        encoding=Encoding.LINEAR,
        source_format="raw",
        source_bit_depth=16,
        metadata={
            "raw_loader": {
                "camera_whitebalance": [2.1, 1.0, 1.5, 1.0],
                "daylight_whitebalance": [2.0, 1.0, 1.4, 1.0],
                "black_level_per_channel": [512, 512, 512, 512],
                "white_level": 16383,
                "color_desc": "RGBG",
                "num_colors": 3,
                "sizes": {"width": 100, "height": 100},
                "demosaic_algorithm": "LMMSE",
                "highlight_mode": "Blend",
                "fbdd_noise_reduction": "Light",
                "exp_shift_applied": 2.0,
                "exp_preserve_highlights": 0.75,
                "used_camera_wb": True,
                "used_auto_wb": False,
            }
        },
    )


@pytest.fixture
def sample_jpg(tmp_path: Path) -> Path:
    """Write a sample JPEG for I/O testing."""
    from PIL import Image

    rng = np.random.default_rng(42)
    arr = rng.random((100, 100, 3), dtype=np.float32)
    img = Image.fromarray((arr * 255).astype(np.uint8))
    path = tmp_path / "test.jpg"
    img.save(path)
    return path


# ── Test image paths ──

TESTS_IMAGE_DIR = Path(__file__).parent / "image"


class _MockSizes:
    """Mock rawpy sizes object."""

    def __init__(self, width: int = 100, height: int = 100) -> None:
        self.width = width
        self.height = height


def make_bayer(
    width: int = 100,
    height: int = 100,
    saturation_ratio: float = 0.0,
    noise_std: float = 0.001,
    black_level: int = 512,
    white_level: int = 16383,
    seed: int = 42,
) -> np.ndarray:
    """Create synthetic Bayer data for testing _analyze_bayer.

    Returns uint16 array of shape (height, width) simulating raw_image_visible.
    The top 20% of rows are dark (near black level) to provide dark-region
    noise measurement. The rest is mid-gray signal.
    """
    rng = np.random.default_rng(seed)
    dynamic_range = white_level - black_level

    bayer = np.empty((height, width), dtype=np.float64)

    # Dark region (top 20%): near black level + noise
    dark_rows = height // 5
    dark_noise = rng.normal(0, noise_std * dynamic_range, (dark_rows, width))
    bayer[:dark_rows] = dark_noise

    # Mid-gray region (bottom 80%)
    mid_rows = height - dark_rows
    mid_noise = rng.normal(0, noise_std * dynamic_range, (mid_rows, width))
    bayer[dark_rows:] = 0.3 * dynamic_range + mid_noise

    # Inject saturated pixels
    if saturation_ratio > 0:
        n_saturated = int(height * width * saturation_ratio)
        indices = rng.choice(height * width, n_saturated, replace=False)
        bayer.flat[indices] = dynamic_range * rng.uniform(0.96, 1.0, n_saturated)

    bayer = np.clip(bayer + black_level, 0, white_level).astype(np.uint16)
    return bayer


class MockRawPy:
    """Mock rawpy.RawPy for unit testing _load_raw without actual RAW files.

    Usage:
        mock_raw = MockRawPy(width=50, height=50, camera_wb=[2.0, 1.0, 1.5, 1.0])
        with patch("rawpy.imread", return_value=mock_raw):
            image = load_image(Path("fake.arw"))
    """

    def __init__(
        self,
        width: int = 100,
        height: int = 100,
        camera_wb: list[float] | None = None,
        daylight_wb: list[float] | None = None,
        black_levels: list[int] | None = None,
        white_level: int = 16383,
        color_desc: bytes = b"RGBG",
        num_colors: int = 3,
        bayer_data: np.ndarray | None = None,
    ) -> None:
        self._width = width
        self._height = height
        self.camera_whitebalance = camera_wb or [2.1, 1.0, 1.5, 1.0]
        self.daylight_whitebalance = daylight_wb or [2.0, 1.0, 1.4, 1.0]
        self.black_level_per_channel = black_levels or [512, 512, 512, 512]
        self.white_level = white_level
        self.color_desc = color_desc
        self.num_colors = num_colors
        self.sizes = _MockSizes(width, height)
        self._last_pp_kwargs: dict[str, Any] = {}
        self._bayer_data = bayer_data

    @property
    def raw_image_visible(self) -> np.ndarray:
        if self._bayer_data is not None:
            return self._bayer_data
        return make_bayer(
            self._width,
            self._height,
            black_level=self.black_level_per_channel[0],
            white_level=self.white_level,
        )

    def postprocess(self, **kwargs: Any) -> np.ndarray:
        self._last_pp_kwargs = kwargs
        rng = np.random.default_rng(42)
        # Return uint16 array matching rawpy output_bps=16
        return (rng.random((self._height, self._width, 3)) * 65535).astype(np.uint16)

    def __enter__(self) -> MockRawPy:
        return self

    def __exit__(self, *args: Any) -> None:
        pass
