"""Image I/O: load and save RAW, HEIF, JPG, PNG, TIFF."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

RAW_EXTENSIONS = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2", ".raf"}
HEIF_EXTENSIONS = {".heif", ".heic"}
JPEG_EXTENSIONS = {".jpg", ".jpeg"}
SUPPORTED_EXTENSIONS = (
    RAW_EXTENSIONS | HEIF_EXTENSIONS | JPEG_EXTENSIONS | {".png", ".tiff", ".tif"}
)


@dataclass
class ImageInfo:
    data: np.ndarray  # (H, W, C) float32 [0.0, 1.0]
    format: str  # "raw", "heif", "jpg", "png", "tiff"
    original_path: Path
    bit_depth: int  # 8 or 16


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


def load_image(path: Path) -> ImageInfo:
    fmt = detect_format(path)
    if fmt == "raw":
        return _load_raw(path)
    if fmt == "heif":
        return _load_heif(path)
    return _load_standard(path, fmt)


def _load_raw(path: Path) -> ImageInfo:
    import rawpy

    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(use_camera_wb=True, output_bps=16, no_auto_bright=True)
        data = rgb.astype(np.float32) / 65535.0
        return ImageInfo(data=data, format="raw", original_path=path, bit_depth=16)


def _load_heif(path: Path) -> ImageInfo:
    import pillow_heif
    from PIL import Image

    pillow_heif.register_heif_opener()
    img = Image.open(path).convert("RGB")
    data = np.array(img, dtype=np.float32) / 255.0
    return ImageInfo(data=data, format="heif", original_path=path, bit_depth=8)


def _load_standard(path: Path, fmt: str) -> ImageInfo:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    data = np.array(img, dtype=np.float32) / 255.0
    return ImageInfo(data=data, format=fmt, original_path=path, bit_depth=8)


def save_image(image: np.ndarray, path: Path, quality: int = 95) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(image * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(clipped)
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        img.save(path, quality=quality)
    else:
        img.save(path)
