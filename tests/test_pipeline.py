"""Tests for pipeline orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

import shotline.processors  # noqa: F401
from shotline.config import PipelineConfig
from shotline.pipeline import Pipeline


def test_pipeline_with_stub_processors(sample_jpg: Path, tmp_path: Path):
    """Pipeline runs with processors that don't need models and match input type."""
    out = tmp_path / "out.jpg"
    config = PipelineConfig(default_steps=["raw_develop", "tone_map", "white_balance"])
    pipeline = Pipeline(config=config)
    result = pipeline.run(sample_jpg, out)

    # raw_develop skipped (jpg not in supported_inputs), tone_map and white_balance run
    assert "raw_develop" in result.skipped
    assert any(s["name"] == "tone_map" for s in result.steps_run)
    assert any(s["name"] == "white_balance" for s in result.steps_run)
    assert out.exists()


def test_pipeline_skips_needs_model(sample_jpg: Path, tmp_path: Path):
    """Pipeline skips processors whose models aren't downloaded."""
    out = tmp_path / "out.jpg"
    config = PipelineConfig(default_steps=["denoise", "white_balance"])
    pipeline = Pipeline(config=config)
    result = pipeline.run(sample_jpg, out)

    assert "denoise" in result.skipped
    assert any(s["name"] == "white_balance" for s in result.steps_run)


def test_pipeline_explicit_steps(sample_jpg: Path, tmp_path: Path):
    """Pipeline respects explicit step list."""
    out = tmp_path / "out.jpg"
    pipeline = Pipeline(steps=["white_balance"])
    result = pipeline.run(sample_jpg, out)

    assert len(result.steps_run) == 1
    assert result.steps_run[0]["name"] == "white_balance"


def test_pipeline_unknown_step_raises():
    with pytest.raises(ValueError, match="Unknown processor"):
        Pipeline(steps=["nonexistent"])


def test_pipeline_output_is_valid_image(sample_jpg: Path, tmp_path: Path):
    out = tmp_path / "out.jpg"
    Pipeline(steps=["white_balance"]).run(sample_jpg, out)

    from PIL import Image

    img = Image.open(out)
    assert img.size == (100, 100)


def test_pipeline_result_to_dict(sample_jpg: Path, tmp_path: Path):
    out = tmp_path / "out.jpg"
    result = Pipeline(steps=["white_balance"]).run(sample_jpg, out)
    d = result.to_dict()
    assert "steps" in d
    assert "skipped" in d
    assert isinstance(d["steps"], list)


def test_pipeline_encoding_in_result(sample_jpg: Path, tmp_path: Path):
    """Pipeline result includes encoding info for each step."""
    out = tmp_path / "out.jpg"
    result = Pipeline(steps=["tone_map", "white_balance"]).run(sample_jpg, out)
    for step in result.steps_run:
        assert "encoding" in step
        assert step["encoding"] in ("linear", "srgb")
