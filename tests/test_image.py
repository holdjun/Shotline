"""Tests for ImageData and sRGB transfer functions."""

from __future__ import annotations

import numpy as np

from shotline.image import (
    Encoding,
    ImageData,
    linear_to_srgb,
    srgb_to_linear,
)


def test_linear_to_srgb_low_values():
    """Low values use the linear segment (x * 12.92)."""
    data = np.array([0.001, 0.003], dtype=np.float32)
    result = linear_to_srgb(data)
    expected = data * 12.92
    np.testing.assert_allclose(result, expected, atol=1e-5)


def test_linear_to_srgb_high_values():
    """Higher values use the power curve."""
    data = np.array([0.5, 1.0], dtype=np.float32)
    result = linear_to_srgb(data)
    assert result[0] > 0.5  # gamma expands midtones
    np.testing.assert_allclose(result[1], 1.0, atol=1e-5)


def test_srgb_to_linear_roundtrip():
    """sRGB → linear → sRGB should be identity."""
    data = np.linspace(0, 1, 256, dtype=np.float32)
    roundtrip = linear_to_srgb(srgb_to_linear(data))
    np.testing.assert_allclose(roundtrip, data, atol=1e-5)


def test_linear_to_srgb_handles_negatives():
    """Negative inputs should be clamped to 0."""
    data = np.array([-0.5, 0.0], dtype=np.float32)
    result = linear_to_srgb(data)
    assert result[0] == 0.0
    assert result[1] == 0.0


def test_image_data_to_linear(sample_image: ImageData):
    assert sample_image.is_srgb
    linear = sample_image.to_linear()
    assert linear.is_linear
    assert linear.data.shape == sample_image.data.shape


def test_image_data_to_srgb_noop(sample_image: ImageData):
    """to_srgb on sRGB image returns self."""
    result = sample_image.to_srgb()
    assert result is sample_image


def test_image_data_to_linear_noop(sample_linear_image: ImageData):
    """to_linear on linear image returns self."""
    result = sample_linear_image.to_linear()
    assert result is sample_linear_image


def test_image_data_to_srgb(sample_linear_image: ImageData):
    srgb = sample_linear_image.to_srgb()
    assert srgb.is_srgb
    assert srgb.data.min() >= 0.0
    assert srgb.data.max() <= 1.0


def test_image_data_replace_merges_metadata(sample_image: ImageData):
    img1 = sample_image.replace(metadata={"step1": "a"})
    img2 = img1.replace(metadata={"step2": "b"})
    assert img2.metadata == {"step1": "a", "step2": "b"}


def test_image_data_properties(sample_image: ImageData):
    assert sample_image.height == 100
    assert sample_image.width == 100


def test_image_data_encoding_values():
    assert Encoding.LINEAR.value == "linear"
    assert Encoding.SRGB.value == "srgb"
