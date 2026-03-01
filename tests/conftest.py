from __future__ import annotations

from pathlib import Path

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
def sample_jpg(tmp_path: Path) -> Path:
    """Write a sample JPEG for I/O testing."""
    from PIL import Image

    rng = np.random.default_rng(42)
    arr = rng.random((100, 100, 3), dtype=np.float32)
    img = Image.fromarray((arr * 255).astype(np.uint8))
    path = tmp_path / "test.jpg"
    img.save(path)
    return path
