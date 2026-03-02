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

    def postprocess(self, **kwargs: Any) -> np.ndarray:
        self._last_pp_kwargs = kwargs
        rng = np.random.default_rng(42)
        # Return uint16 array matching rawpy output_bps=16
        return (rng.random((self._height, self._width, 3)) * 65535).astype(np.uint16)

    def __enter__(self) -> MockRawPy:
        return self

    def __exit__(self, *args: Any) -> None:
        pass
