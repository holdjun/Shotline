"""Image I/O: load and save RAW, HEIF, JPG, PNG, TIFF."""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from shotline.image import Encoding, ImageData

logger = logging.getLogger(__name__)

RAW_EXTENSIONS = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2", ".raf"}
HEIF_EXTENSIONS = {".heif", ".heic", ".hif"}
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


def load_image(path: Path, raw_params: dict[str, Any] | None = None) -> ImageData:
    fmt = detect_format(path)
    if fmt == "raw":
        return _load_raw(path, raw_params)
    if fmt == "heif":
        return _load_heif(path)
    return _load_standard(path, fmt)


# ── RAW enum mappings ──

_DEMOSAIC_MAP: dict[str, int] = {}
_HIGHLIGHT_MAP: dict[str, int] = {}
_FBDD_MAP: dict[str, int] = {}


def _init_rawpy_maps() -> None:
    """Lazily populate enum maps on first RAW load."""
    if _DEMOSAIC_MAP:
        return
    import rawpy

    for member in rawpy.DemosaicAlgorithm:
        _DEMOSAIC_MAP[member.name.upper()] = member.value
    for member in rawpy.HighlightMode:
        _HIGHLIGHT_MAP[member.name.upper()] = member.value
    for member in rawpy.FBDDNoiseReductionMode:
        _FBDD_MAP[member.name.upper()] = member.value


def _resolve_demosaic(name: str) -> tuple[int, str]:
    """Resolve demosaic name, falling back to DHT if algorithm needs GPL pack."""
    import rawpy

    value = _DEMOSAIC_MAP[name.upper()]
    try:
        rawpy.DemosaicAlgorithm(value).checkSupported()
        return value, name.upper()
    except rawpy._rawpy.NotSupportedError:
        fallback = _DEMOSAIC_MAP["DHT"]
        return fallback, "DHT"


def _resolve_highlight(value: str | int) -> int:
    if isinstance(value, int):
        return value  # libraw reconstruct levels 3-9
    return _HIGHLIGHT_MAP[value.upper()]


def _resolve_fbdd(name: str) -> int:
    return _FBDD_MAP[name.upper()]


def _analyze_bayer(
    raw: Any,
    *,
    saturation_threshold: float = 0.005,
    noise_low: float = 0.003,
    noise_high: float = 0.008,
) -> dict[str, Any]:
    """Analyze raw Bayer data before postprocess to guide adaptive decisions.

    Thresholds are configurable via TOML [processor_params.raw_develop]:
      - saturation_threshold: Bayer pixel saturation ratio to trigger Blend (default 0.005)
      - noise_low: below this → FBDD Off (default 0.003)
      - noise_high: above this → FBDD Full; between low/high → FBDD Light (default 0.008)
    """
    bayer = raw.raw_image_visible.astype(np.float32)
    black_level = float(np.mean(raw.black_level_per_channel))
    white_level = float(raw.white_level)
    dynamic_range = white_level - black_level

    normalized = (bayer - black_level) / dynamic_range

    saturation_ratio = float((normalized >= 0.95).mean())

    # Noise from dark region (bottom 5% of signal)
    dark_pixels = normalized[normalized < 0.05]
    noise_std = float(dark_pixels.std()) if dark_pixels.size > 100 else 0.0

    # Adaptive decisions using configurable thresholds
    recommended_highlight = "Blend" if saturation_ratio >= saturation_threshold else "Clip"
    if noise_std < noise_low:
        recommended_fbdd = "Off"
    elif noise_std < noise_high:
        recommended_fbdd = "Light"
    else:
        recommended_fbdd = "Full"

    return {
        "saturation_ratio": saturation_ratio,
        "noise_std": noise_std,
        "recommended_highlight": recommended_highlight,
        "recommended_fbdd": recommended_fbdd,
    }


def _compute_wb_compensation(camera_wb: list[float]) -> float:
    """Compute brightness compensation factor for Blend highlight mode.

    Blend mode divides all channels by max(WB) to prevent overflow.
    This factor restores mid-tone brightness while preserving HDR headroom.
    """
    wb_rgb = [v for v in camera_wb[:3] if v > 0]
    if len(wb_rgb) < 2:
        return 1.0
    return max(wb_rgb) / min(wb_rgb)


def _extract_raw_metadata(
    raw: Any,
    *,
    demosaic_algorithm: str,
    highlight_mode: str | int,
    fbdd_noise_reduction: str,
    exp_shift_applied: float,
    exp_preserve_highlights: float,
    used_camera_wb: bool,
    used_auto_wb: bool,
    bayer_analysis: dict[str, Any] | None = None,
    wb_compensation: float = 1.0,
) -> dict[str, Any]:
    """Extract metadata from rawpy handle + processing params."""
    sizes = raw.sizes
    meta: dict[str, Any] = {
        "camera_whitebalance": list(raw.camera_whitebalance),
        "daylight_whitebalance": list(raw.daylight_whitebalance),
        "black_level_per_channel": list(raw.black_level_per_channel),
        "white_level": int(raw.white_level),
        "color_desc": (
            raw.color_desc.decode() if isinstance(raw.color_desc, bytes) else str(raw.color_desc)
        ),
        "num_colors": int(raw.num_colors),
        "sizes": {"width": sizes.width, "height": sizes.height},
        "demosaic_algorithm": str(demosaic_algorithm),
        "highlight_mode": str(highlight_mode),
        "fbdd_noise_reduction": str(fbdd_noise_reduction),
        "exp_shift_applied": exp_shift_applied,
        "exp_preserve_highlights": exp_preserve_highlights,
        "used_camera_wb": used_camera_wb,
        "used_auto_wb": used_auto_wb,
        "wb_compensation": wb_compensation,
    }
    if bayer_analysis is not None:
        meta["bayer_analysis"] = bayer_analysis
    return meta


def _extract_exif(path: Path, raw: Any = None) -> dict[str, Any]:
    """Extract camera/lens EXIF from a RAW file.

    Primary: exifread on the file directly.
    Fallback: rawpy extract_thumb → PIL EXIF.
    Returns empty dict on failure.
    """
    exif: dict[str, Any] = {}

    # Primary path: exifread
    try:
        import exifread

        with open(path, "rb") as f:
            tags = exifread.process_file(f, stop_tag="UNDEF", details=False)

        def _tag_str(key: str) -> str | None:
            v = tags.get(key)
            return str(v).strip() if v else None

        def _tag_float(key: str) -> float | None:
            v = tags.get(key)
            if v is None:
                return None
            # exifread returns Ratio objects; str(v) gives e.g. "24" or "28/10"
            s = str(v).strip()
            if "/" in s:
                num, den = s.split("/", 1)
                return float(num) / float(den) if float(den) != 0 else None
            try:
                return float(s)
            except ValueError:
                return None

        exif = {
            "camera_make": _tag_str("Image Make"),
            "camera_model": _tag_str("Image Model"),
            "lens_make": _tag_str("EXIF LensMake"),
            "lens_model": _tag_str("EXIF LensModel"),
            "focal_length": _tag_float("EXIF FocalLength"),
            "aperture": _tag_float("EXIF FNumber"),
            "iso": _tag_float("EXIF ISOSpeedRatings"),
        }
        # Remove None values
        exif = {k: v for k, v in exif.items() if v is not None}
        if exif.get("camera_make") and exif.get("camera_model"):
            return exif
    except ImportError:
        pass
    except Exception:
        logger.debug("exifread failed for %s, trying PIL fallback", path)

    # Fallback: rawpy thumbnail → PIL EXIF
    if raw is not None:
        try:
            from PIL import Image
            from PIL.ExifTags import Base as ExifBase

            thumb = raw.extract_thumb()
            if thumb.format == 1:  # JPEG
                img = Image.open(BytesIO(thumb.data))
                pil_exif = img.getexif()
                exif_ifd = pil_exif.get_ifd(0x8769)  # Exif IFD

                exif = {
                    "camera_make": pil_exif.get(ExifBase.Make),
                    "camera_model": pil_exif.get(ExifBase.Model),
                    "lens_make": exif_ifd.get(42035),  # LensMake
                    "lens_model": exif_ifd.get(42036),  # LensModel
                    "focal_length": float(exif_ifd.get(37386, 0)) or None,
                    "aperture": float(exif_ifd.get(33437, 0)) or None,
                    "iso": exif_ifd.get(34855),
                }
                exif = {k: v for k, v in exif.items() if v is not None}
        except Exception:
            logger.debug("PIL EXIF fallback failed for %s", path)

    return exif


def _load_raw(path: Path, raw_params: dict[str, Any] | None = None) -> ImageData:
    """Load RAW as LINEAR float32 with adaptive rawpy parameters.

    Performs Bayer analysis before postprocess to auto-select highlight mode
    and FBDD noise reduction. User-specified values in raw_params override
    the adaptive choices.

    After Blend highlight mode, applies WB compensation to restore brightness
    while preserving HDR headroom (values > 1.0 in highlights).
    """
    import rawpy

    _init_rawpy_maps()

    params = raw_params or {}
    demosaic_requested = params.get("demosaic_algorithm", "DHT")
    ev = float(params.get("ev", 0.0))
    exp_preserve = float(params.get("exp_preserve_highlights", 0.75))
    chromatic_aberration = params.get("chromatic_aberration")
    noise_thr = params.get("noise_thr")
    median_passes = int(params.get("median_filter_passes", 0))

    # User overrides: None = auto-select, explicit string = forced
    highlight_override = params.get("highlight_mode")
    fbdd_override = params.get("fbdd_noise_reduction")

    exp_shift = 2.0**ev if ev != 0.0 else 1.0
    demosaic_value, demosaic_alg = _resolve_demosaic(demosaic_requested)

    with rawpy.imread(str(path)) as raw:
        # ── Bayer analysis (before postprocess) ──
        bayer_analysis = _analyze_bayer(
            raw,
            saturation_threshold=float(params.get("saturation_threshold", 0.005)),
            noise_low=float(params.get("noise_low", 0.003)),
            noise_high=float(params.get("noise_high", 0.008)),
        )

        # Adaptive or user-forced selection
        highlight = highlight_override or bayer_analysis["recommended_highlight"]
        fbdd = fbdd_override or bayer_analysis["recommended_fbdd"]

        # White balance fallback: camera WB → auto WB
        camera_wb = list(raw.camera_whitebalance)
        use_camera_wb = any(v != 0.0 for v in camera_wb)
        use_auto_wb = not use_camera_wb

        pp_kwargs: dict[str, Any] = {
            "gamma": (1, 1),
            "no_auto_bright": True,
            "output_bps": 16,
            "output_color": rawpy.ColorSpace.sRGB,
            "use_camera_wb": use_camera_wb,
            "use_auto_wb": use_auto_wb,
            "demosaic_algorithm": rawpy.DemosaicAlgorithm(demosaic_value),
            "highlight_mode": rawpy.HighlightMode(_resolve_highlight(highlight)),
            "fbdd_noise_reduction": rawpy.FBDDNoiseReductionMode(_resolve_fbdd(fbdd)),
            "exp_shift": exp_shift,
            "exp_preserve_highlights": exp_preserve,
            "median_filter_passes": median_passes,
        }

        if chromatic_aberration is not None:
            pp_kwargs["chromatic_aberration"] = tuple(chromatic_aberration)
        if noise_thr is not None:
            pp_kwargs["noise_thr"] = float(noise_thr)

        rgb = raw.postprocess(**pp_kwargs)
        data = rgb.astype(np.float32) / 65535.0

        # ── WB compensation for Blend mode ──
        wb_compensation = 1.0
        if str(highlight).upper() == "BLEND":
            wb_compensation = _compute_wb_compensation(camera_wb)
            data = data * wb_compensation

        metadata = _extract_raw_metadata(
            raw,
            demosaic_algorithm=demosaic_alg,
            highlight_mode=highlight,
            fbdd_noise_reduction=fbdd,
            exp_shift_applied=exp_shift,
            exp_preserve_highlights=exp_preserve,
            used_camera_wb=use_camera_wb,
            used_auto_wb=use_auto_wb,
            bayer_analysis=bayer_analysis,
            wb_compensation=wb_compensation,
        )

        # ── EXIF extraction (camera/lens identity for lens correction) ──
        exif = _extract_exif(path, raw)
        if exif:
            metadata["exif"] = exif

        return ImageData(
            data=data,
            encoding=Encoding.LINEAR,
            source_format="raw",
            source_bit_depth=16,
            original_path=path,
            metadata={"raw_loader": metadata},
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
