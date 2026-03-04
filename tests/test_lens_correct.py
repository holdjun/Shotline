"""Tests for lens correction processor."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import shotline.processors  # noqa: F401
from shotline.image import Encoding, ImageData
from shotline.processor import ProcessorStatus, get_processor, list_processors
from shotline.processors.lens_correct import _find_repo_db_files, _get_database
from tests.conftest import SAMPLE_EXIF

try:
    import lensfunpy  # noqa: F401

    _skip_no_lensfunpy = False
except ImportError:
    _skip_no_lensfunpy = True

_requires_lensfunpy = pytest.mark.skipif(_skip_no_lensfunpy, reason="lensfunpy not installed")


def _make_linear_with_exif(
    width: int = 100,
    height: int = 100,
    exif: dict | None = None,
) -> ImageData:
    """Create a LINEAR image with EXIF metadata for lens correction testing."""
    rng = np.random.default_rng(42)
    data = rng.random((height, width, 3), dtype=np.float32) * 1.5
    return ImageData(
        data=data,
        encoding=Encoding.LINEAR,
        source_format="raw",
        source_bit_depth=16,
        metadata={"raw_loader": {"exif": exif or dict(SAMPLE_EXIF)}},
    )


# ── Registration and meta ──


def test_lens_correct_registered():
    metas = list_processors()
    names = [m.name for m in metas]
    assert "lens_correct" in names


def test_lens_correct_meta():
    proc = get_processor("lens_correct")
    meta = proc.meta()
    assert meta.name == "lens_correct"
    assert meta.order == 5
    assert meta.supported_inputs == ["raw"]


def test_lens_correct_order_before_raw_develop():
    metas = list_processors()
    orders = {m.name: m.order for m in metas}
    assert orders["lens_correct"] < orders["raw_develop"]


# ── Status ──


@_requires_lensfunpy
def test_lens_correct_status_available():
    proc = get_processor("lens_correct")
    assert proc.status() == ProcessorStatus.AVAILABLE


def test_lens_correct_status_unavailable():
    proc = get_processor("lens_correct")
    with patch("shotline.processors.lens_correct._has_lensfunpy", return_value=False):
        assert proc.status() == ProcessorStatus.UNAVAILABLE


# ── Skips without required data ──


def test_lens_correct_skips_without_exif():
    """No EXIF → graceful skip."""
    image = ImageData(
        data=np.zeros((50, 50, 3), dtype=np.float32),
        encoding=Encoding.LINEAR,
        source_format="raw",
        source_bit_depth=16,
        metadata={"raw_loader": {}},
    )
    proc = get_processor("lens_correct")
    result = proc.process(image)
    assert "lens_correct" in result.metadata
    assert "skipped" in result.metadata["lens_correct"]


def test_lens_correct_skips_without_camera_model():
    """EXIF without camera_model → skip."""
    image = _make_linear_with_exif(exif={"camera_make": "Sony"})
    proc = get_processor("lens_correct")
    result = proc.process(image)
    assert "skipped" in result.metadata["lens_correct"]


@_requires_lensfunpy
def test_lens_correct_skips_without_lens_model():
    """EXIF without lens_model → skip (can't look up lens)."""
    image = _make_linear_with_exif(
        exif={
            "camera_make": "Sony",
            "camera_model": "ILCE-7M3",
            "focal_length": 35.0,
            "aperture": 2.8,
        }
    )
    proc = get_processor("lens_correct")
    result = proc.process(image)
    assert "skipped" in result.metadata["lens_correct"]


@_requires_lensfunpy
def test_lens_correct_skips_unknown_camera():
    """Unknown camera → skip with info."""
    image = _make_linear_with_exif(
        exif={
            "camera_make": "FakeManufacturer",
            "camera_model": "NoSuchCamera9000",
            "lens_model": "NoSuchLens 50mm f/1.4",
            "focal_length": 50.0,
            "aperture": 1.4,
        }
    )
    proc = get_processor("lens_correct")
    result = proc.process(image)
    assert "skipped" in result.metadata["lens_correct"]


@_requires_lensfunpy
def test_lens_correct_skips_missing_focal_or_aperture():
    """EXIF without focal_length or aperture → skip."""
    image = _make_linear_with_exif(
        exif={
            "camera_make": "Sony",
            "camera_model": "ILCE-7M3",
            "lens_model": "FE 24-70mm F2.8 GM",
            # no focal_length or aperture
        }
    )
    proc = get_processor("lens_correct")
    result = proc.process(image)
    assert "skipped" in result.metadata["lens_correct"]


# ── Correction application ──


@_requires_lensfunpy
def test_lens_correct_applies_with_valid_exif():
    """With valid EXIF for a known camera/lens, corrections are applied."""
    image = _make_linear_with_exif()
    proc = get_processor("lens_correct")
    result = proc.process(image)

    meta = result.metadata["lens_correct"]
    # Should not be skipped (Sony A7III + FE 24-70 GM is in lensfun DB)
    assert "skipped" not in meta
    assert "corrections" in meta
    assert meta["focal_length"] == 35.0
    assert meta["aperture"] == 2.8


@_requires_lensfunpy
def test_lens_correct_preserves_encoding():
    """Output stays LINEAR."""
    image = _make_linear_with_exif()
    proc = get_processor("lens_correct")
    result = proc.process(image)
    assert result.encoding == Encoding.LINEAR


@_requires_lensfunpy
def test_lens_correct_output_shape():
    """Output may be slightly smaller due to auto-crop of distortion black borders."""
    image = _make_linear_with_exif()
    proc = get_processor("lens_correct")
    result = proc.process(image)
    # Auto-crop may reduce size, but should not increase it
    assert result.data.shape[0] <= image.data.shape[0]
    assert result.data.shape[1] <= image.data.shape[1]
    assert result.data.shape[2] == 3


@_requires_lensfunpy
def test_lens_correct_modifies_data():
    """Corrections should change pixel values."""
    image = _make_linear_with_exif()
    proc = get_processor("lens_correct")
    result = proc.process(image)

    # At least some pixels should be different
    if "skipped" not in result.metadata.get("lens_correct", {}):
        assert not np.array_equal(result.data, image.data)


@_requires_lensfunpy
def test_lens_correct_metadata_fields():
    """Verify all expected metadata fields are present."""
    image = _make_linear_with_exif()
    proc = get_processor("lens_correct")
    result = proc.process(image)

    meta = result.metadata["lens_correct"]
    if "skipped" not in meta:
        assert "camera" in meta
        assert "lens" in meta
        assert "crop_factor" in meta
        assert "focal_length" in meta
        assert "aperture" in meta
        assert "distance" in meta
        assert "corrections" in meta


# ── Config params ──


@_requires_lensfunpy
def test_lens_correct_disable_distortion():
    """correct_distortion=False with TCA still applies distortion (TCA path includes it)."""
    image = _make_linear_with_exif()
    proc = get_processor("lens_correct")
    # With TCA enabled (default), distortion is always included via subpixel_geometry_distortion
    result = proc.process(image, {"correct_distortion": False})
    meta = result.metadata["lens_correct"]
    if "corrections" in meta:
        # TCA path forces distortion on
        assert meta["corrections"].get("tca") is True
        assert meta["corrections"].get("distortion") is True


@_requires_lensfunpy
def test_lens_correct_distortion_only_without_tca():
    """correct_tca=False → single-channel remap, no TCA in output."""
    image = _make_linear_with_exif()
    proc = get_processor("lens_correct")
    result = proc.process(image, {"correct_tca": False})
    meta = result.metadata["lens_correct"]
    if "corrections" in meta:
        assert meta["corrections"].get("distortion") is True
        assert "tca" not in meta["corrections"]


@_requires_lensfunpy
def test_lens_correct_no_geometry_without_tca_and_distortion():
    """Both TCA and distortion disabled → only vignetting."""
    image = _make_linear_with_exif()
    proc = get_processor("lens_correct")
    result = proc.process(image, {"correct_tca": False, "correct_distortion": False})
    meta = result.metadata["lens_correct"]
    if "corrections" in meta:
        assert "tca" not in meta["corrections"]
        assert "distortion" not in meta["corrections"]


@_requires_lensfunpy
def test_lens_correct_disable_vignetting():
    """correct_vignetting=False skips vignetting."""
    image = _make_linear_with_exif()
    proc = get_processor("lens_correct")
    result = proc.process(image, {"correct_vignetting": False})
    meta = result.metadata["lens_correct"]
    if "corrections" in meta:
        assert "vignetting" not in meta["corrections"] or not meta["corrections"].get("vignetting")


@_requires_lensfunpy
def test_lens_correct_custom_distance():
    """Custom distance parameter is recorded in metadata."""
    image = _make_linear_with_exif()
    proc = get_processor("lens_correct")
    result = proc.process(image, {"distance": 5.0})
    meta = result.metadata["lens_correct"]
    if "skipped" not in meta:
        assert meta["distance"] == 5.0


# ── Vignetting-only without cv2 ──


@_requires_lensfunpy
def test_lens_correct_vignetting_only_without_cv2():
    """Without cv2, only vignetting correction is applied."""
    image = _make_linear_with_exif()
    proc = get_processor("lens_correct")
    with patch("shotline.processors.lens_correct._has_cv2", return_value=False):
        result = proc.process(image)

    meta = result.metadata["lens_correct"]
    if "corrections" in meta:
        corrections = meta["corrections"]
        # No geometric corrections without cv2
        assert "distortion" not in corrections
        assert "tca" not in corrections


# ── Database caching and repo DB loading ──


@_requires_lensfunpy
def test_get_database_caching():
    """Database instance is cached — second call returns same object."""
    import shotline.processors.lens_correct as mod

    mod._db_cache = None
    try:
        db1 = _get_database()
        db2 = _get_database()
        assert db1 is db2
    finally:
        mod._db_cache = None


def test_find_repo_db_files_returns_list():
    """_find_repo_db_files returns a list (possibly empty if no XML yet)."""
    result = _find_repo_db_files()
    assert isinstance(result, list)
    for path in result:
        assert path.endswith(".xml")


def test_find_repo_db_files_walks_to_project_root(tmp_path: Path):
    """_find_repo_db_files walks up to pyproject.toml and finds XML files."""
    fake_root = tmp_path / "project"
    fake_root.mkdir()
    (fake_root / "pyproject.toml").touch()
    db_dir = fake_root / "data" / "lensfun-db"
    db_dir.mkdir(parents=True)
    (db_dir / "mil-sony.xml").write_text("<lensdatabase/>")
    (db_dir / "compact-nikon.xml").write_text("<lensdatabase/>")

    fake_module_dir = fake_root / "src" / "shotline" / "processors"
    fake_module_dir.mkdir(parents=True)

    real_resolve = Path.resolve

    def patched_resolve(self: Path) -> Path:
        if self.name == "lens_correct.py" and "shotline" in str(self):
            return fake_module_dir / "lens_correct.py"
        return real_resolve(self)

    with patch.object(Path, "resolve", patched_resolve):
        result = _find_repo_db_files()

    assert len(result) == 2
    assert all(p.endswith(".xml") for p in result)


@_requires_lensfunpy
def test_database_loads_extra_paths():
    """Database with extra paths should have >= cameras/lenses as bundled."""
    import lensfunpy

    import shotline.processors.lens_correct as mod

    bundled_db = lensfunpy.Database()
    bundled_cams = len(bundled_db.cameras)
    bundled_lenses = len(bundled_db.lenses)

    mod._db_cache = None
    try:
        db = _get_database()
        assert len(db.cameras) >= bundled_cams
        assert len(db.lenses) >= bundled_lenses
    finally:
        mod._db_cache = None
