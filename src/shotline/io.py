"""Image I/O: load and save RAW, HEIF, JPG, PNG, TIFF."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from shotline.image import Encoding, ImageData

RAW_EXTENSIONS = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2", ".raf"}
HEIF_EXTENSIONS = {".heif", ".heic"}
JPEG_EXTENSIONS = {".jpg", ".jpeg"}
SUPPORTED_EXTENSIONS = (
    RAW_EXTENSIONS | HEIF_EXTENSIONS | JPEG_EXTENSIONS | {".png", ".tiff", ".tif"}
)


def detect_format(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in RAW_EXTENSIONS:
        return "raw"
    if ext in HEIF_EXTENSIONS:
        return "heif"
    if ext in JPEG_EXTENSIONS:
        return "jpg"
    if ext in {".png"}:
        return "png"
    if ext in {".tiff", ".tif"}:
        return "tiff"
    raise ValueError(f"Unsupported format: {ext}")


def load_image(path: Path) -> ImageData:
    fmt = detect_format(path)
    if fmt == "raw":
        return _load_raw(path)
    if fmt == "heif":
        return _load_heif(path)
    return _load_standard(path, fmt)


def _load_raw(path: Path) -> ImageData:
    """Load RAW as LINEAR float32. gamma=(1,1) skips sRGB gamma."""
    import rawpy

    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            gamma=(1, 1),
            no_auto_bright=True,
            use_camera_wb=True,
            output_bps=16,
        )
        data = rgb.astype(np.float32) / 65535.0
        return ImageData(
            data=data,
            encoding=Encoding.LINEAR,
            source_format="raw",
            source_bit_depth=16,
            original_path=path,
        )


def _load_heif(path: Path) -> ImageData:
    """Load HEIF/HEIC as sRGB. Detects 10-bit iPhone photos."""
    import pillow_heif
    from PIL import Image

    pillow_heif.register_heif_opener()
    img = Image.open(path)
    bit_depth = 10 if img.mode in ("I;16", "I;16L", "I;16B") else 8
    img = img.convert("RGB")
    data = np.array(img, dtype=np.float32) / 255.0
    return ImageData(
        data=data,
        encoding=Encoding.SRGB,
        source_format="heif",
        source_bit_depth=bit_depth,
        original_path=path,
    )


def _load_standard(path: Path, fmt: str) -> ImageData:
    """Load JPG/PNG/TIFF as sRGB. Supports 16-bit PNG/TIFF."""
    from PIL import Image

    img = Image.open(path)

    if img.mode in ("I;16", "I;16L", "I;16B", "I"):
        bit_depth = 16
        arr = np.array(img, dtype=np.float32)
        arr = arr / 65535.0 if arr.max() > 255 else arr / 255.0
        data = np.stack([arr] * 3, axis=-1) if arr.ndim == 2 else arr
    else:
        bit_depth = 8
        img = img.convert("RGB")
        data = np.array(img, dtype=np.float32) / 255.0

    return ImageData(
        data=data.astype(np.float32),
        encoding=Encoding.SRGB,
        source_format=fmt,
        source_bit_depth=bit_depth,
        original_path=path,
    )


def save_image(
    image: ImageData,
    path: Path,
    quality: int = 95,
) -> None:
    """Save ImageData to disk. Auto-converts LINEAR to sRGB.

    Currently saves 8-bit RGB. 16-bit TIFF/PNG support to be added later.
    """
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)

    if image.is_linear:
        image = image.to_srgb()

    ext = path.suffix.lower()
    clipped = np.clip(image.data * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(clipped)

    if ext in {".jpg", ".jpeg"}:
        img.save(path, quality=quality)
    else:
        img.save(path)
