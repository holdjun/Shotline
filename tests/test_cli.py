"""Tests for CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from shotline.cli import app

runner = CliRunner()


def test_list_command():
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "RAW Development" in result.output
    assert "AI Denoise" in result.output
    assert "Exposure Adjust" in result.output


def test_list_json():
    result = runner.invoke(app, ["list", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert len(data) == 10
    names = [p["name"] for p in data]
    assert "raw_develop" in names
    assert "exposure_adjust" in names
    assert "denoise" in names


def test_run_single_image(sample_jpg: Path, tmp_path: Path):
    out = tmp_path / "out.jpg"
    result = runner.invoke(app, ["run", str(sample_jpg), "-o", str(out)])
    assert result.exit_code == 0
    assert out.exists()


def test_run_json_output(sample_jpg: Path, tmp_path: Path):
    out = tmp_path / "out.jpg"
    result = runner.invoke(app, ["run", str(sample_jpg), "-o", str(out), "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert len(data) == 1
    assert "output" in data[0]


def test_run_with_steps(sample_jpg: Path, tmp_path: Path):
    out = tmp_path / "out.jpg"
    result = runner.invoke(app, ["run", str(sample_jpg), "-o", str(out), "-s", "white_balance"])
    assert result.exit_code == 0
    assert out.exists()


def test_run_directory(sample_jpg: Path, tmp_path: Path):
    import numpy as np
    from PIL import Image

    img_dir = tmp_path / "input"
    img_dir.mkdir()

    for i in range(3):
        data = np.random.default_rng(i).integers(0, 255, (50, 50, 3), dtype=np.uint8)
        img = Image.fromarray(data)
        img.save(img_dir / f"img{i}.jpg")

    out_dir = tmp_path / "output"
    result = runner.invoke(app, ["run", str(img_dir), "-o", str(out_dir)])
    assert result.exit_code == 0
    assert out_dir.exists()
    assert len(list(out_dir.iterdir())) == 3


def test_no_args_shows_help():
    result = runner.invoke(app, [])
    assert "Photo processing pipeline" in result.output
