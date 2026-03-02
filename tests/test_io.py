"""Tests for image I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shotline.image import Encoding, ImageData
from shotline.io import (
    SUPPORTED_EXTENSIONS,
    detect_format,
    load_image,
    save_image,
)


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
    assert ".png" in SUPPORTED_EXTENSIONS
