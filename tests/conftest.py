from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def sample_image() -> np.ndarray:
    """100x100 RGB float32 [0,1] test image."""
    rng = np.random.default_rng(42)
    return rng.random((100, 100, 3), dtype=np.float32)


@pytest.fixture
def sample_jpg(tmp_path: Path, sample_image: np.ndarray) -> Path:
    """Write a sample JPEG for I/O testing."""
    from PIL import Image

    img = Image.fromarray((sample_image * 255).astype(np.uint8))
    path = tmp_path / "test.jpg"
    img.save(path)
    return path
