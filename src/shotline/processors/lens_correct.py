"""Lens correction: distortion, vignetting, TCA via lensfunpy."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from shotline.image import ImageData
from shotline.processor import BaseProcessor, ProcessorMeta, ProcessorStatus, register_processor

logger = logging.getLogger(__name__)


def _has_lensfunpy() -> bool:
    try:
        import lensfunpy  # noqa: F401

        return True
    except ImportError:
        return False


def _has_cv2() -> bool:
    try:
        import cv2  # noqa: F401

        return True
    except ImportError:
        return False


def _find_camera_and_lens(
    db: Any,
    camera_make: str,
    camera_model: str,
    lens_model: str | None,
    lens_make: str | None,
) -> tuple[Any, Any] | None:
    """Look up camera and lens in lensfun database.

    Returns (camera, lens) tuple or None if not found.
    """
    cameras = db.find_cameras(camera_make, camera_model)
    if not cameras:
        # Try loose search as fallback
        cameras = db.find_cameras(camera_make, camera_model, loose_search=True)
    if not cameras:
        return None
    cam = cameras[0]

    if not lens_model:
        return None

    lenses = db.find_lenses(cam, lens_make or "", lens_model)
    if not lenses:
        # Try without lens_make for broader matching
        lenses = db.find_lenses(cam, "", lens_model)
    if not lenses:
        # Try loose search
        lenses = db.find_lenses(cam, "", lens_model, loose_search=True)
    if not lenses:
        return None

    return cam, lenses[0]


def _auto_crop_black_border(
    data: np.ndarray, coords: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, dict[str, int]]:
    """Crop to the largest inscribed rectangle with no out-of-bounds pixels.

    After geometric distortion correction, edges may map outside the source
    image, producing black borders. This finds the tightest crop that removes
    all such pixels.

    coords can be shape (h,w,2) for single-channel or (h,w,3,2) for per-channel.
    Returns (cropped_data, crop_info).
    """
    if coords.ndim == 4:
        # Per-channel coords (h, w, 3, 2): pixel is valid if ALL channels are in-bounds
        in_bounds = np.ones(coords.shape[:2], dtype=bool)
        for ch in range(coords.shape[2]):
            in_bounds &= (
                (coords[:, :, ch, 0] >= 0)
                & (coords[:, :, ch, 0] < width)
                & (coords[:, :, ch, 1] >= 0)
                & (coords[:, :, ch, 1] < height)
            )
    else:
        # Single coords (h, w, 2)
        in_bounds = (
            (coords[:, :, 0] >= 0)
            & (coords[:, :, 0] < width)
            & (coords[:, :, 1] >= 0)
            & (coords[:, :, 1] < height)
        )

    # Find rows/cols where ALL pixels are in-bounds
    row_valid = in_bounds.all(axis=1)
    col_valid = in_bounds.all(axis=0)

    if not row_valid.any() or not col_valid.any():
        # Fallback: find rows/cols where >95% pixels are valid
        row_valid = in_bounds.mean(axis=1) > 0.95
        col_valid = in_bounds.mean(axis=0) > 0.95

    if not row_valid.any() or not col_valid.any():
        return data, {}

    valid_rows = np.where(row_valid)[0]
    valid_cols = np.where(col_valid)[0]
    r0, r1 = int(valid_rows[0]), int(valid_rows[-1]) + 1
    c0, c1 = int(valid_cols[0]), int(valid_cols[-1]) + 1

    crop_info = {"top": r0, "bottom": data.shape[0] - r1, "left": c0, "right": data.shape[1] - c1}

    # Only crop if there's actually a meaningful border
    if r0 == 0 and r1 == data.shape[0] and c0 == 0 and c1 == data.shape[1]:
        return data, {}

    return data[r0:r1, c0:c1].copy(), crop_info


def _apply_corrections(
    data: np.ndarray,
    cam: Any,
    lens: Any,
    focal_length: float,
    aperture: float,
    distance: float,
    *,
    correct_distortion: bool = True,
    correct_vignetting: bool = True,
    correct_tca: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply lensfunpy corrections to LINEAR float32 image data.

    Order: vignetting first (modifies pixel values in-place),
    then TCA + distortion (geometric remaps via cv2),
    then auto-crop to remove black borders from geometric correction.
    """
    import lensfunpy

    height, width = data.shape[:2]
    applied: dict[str, Any] = {}
    has_cv2 = _has_cv2()

    # Determine which corrections to enable
    flags = 0
    if correct_vignetting:
        flags |= lensfunpy.ModifyFlags.VIGNETTING
    if correct_tca and has_cv2:
        flags |= lensfunpy.ModifyFlags.TCA
    if correct_distortion and has_cv2:
        flags |= lensfunpy.ModifyFlags.DISTORTION

    if flags == 0:
        return data, {"skipped": "no corrections enabled or cv2 missing for geometry"}

    mod = lensfunpy.Modifier(lens, cam.crop_factor, width, height)
    mod.initialize(
        focal_length,
        aperture,
        distance,
        pixel_format=np.float32,
        flags=flags,
    )

    result = data.copy()

    # 1. Vignetting correction (in-place pixel value modification)
    if correct_vignetting and (flags & lensfunpy.ModifyFlags.VIGNETTING):
        did_apply = mod.apply_color_modification(result)
        applied["vignetting"] = bool(did_apply)

    # 2. Geometry corrections (TCA + distortion via cv2.remap)
    need_geometry = (correct_tca or correct_distortion) and has_cv2
    remap_coords = None
    if need_geometry:
        import cv2

        if correct_tca:
            # apply_subpixel_geometry_distortion combines TCA + distortion
            if correct_distortion:
                remap_coords = mod.apply_subpixel_geometry_distortion()
            else:
                remap_coords = mod.apply_subpixel_distortion()

            if remap_coords is not None:
                for ch in range(3):
                    result[..., ch] = cv2.remap(
                        result[..., ch],
                        remap_coords[..., ch, :],
                        None,
                        cv2.INTER_LANCZOS4,
                    )
                applied["tca"] = True
                if correct_distortion:
                    applied["distortion"] = True
        elif correct_distortion:
            # Distortion only (no TCA)
            remap_coords = mod.apply_geometry_distortion()
            if remap_coords is not None:
                result = cv2.remap(result, remap_coords, None, cv2.INTER_LANCZOS4)
                applied["distortion"] = True

    # 3. Auto-crop black borders from geometric correction
    if remap_coords is not None:
        result, crop_info = _auto_crop_black_border(result, remap_coords, width, height)
        if crop_info:
            applied["auto_crop"] = crop_info

    return result, applied


@register_processor
class LensCorrectProcessor(BaseProcessor):
    def meta(self) -> ProcessorMeta:
        return ProcessorMeta(
            name="lens_correct",
            display_name="Lens Correction",
            description="Distortion, vignetting, and TCA correction via lensfunpy",
            order=5,
            supported_inputs=["raw"],
        )

    def status(self) -> ProcessorStatus:
        if not _has_lensfunpy():
            return ProcessorStatus.UNAVAILABLE
        return ProcessorStatus.AVAILABLE

    def process(self, image: ImageData, params: dict[str, Any] | None = None) -> ImageData:
        params = params or {}

        # Read EXIF from raw_loader metadata
        exif = image.metadata.get("raw_loader", {}).get("exif", {})
        camera_make = exif.get("camera_make")
        camera_model = exif.get("camera_model")

        if not camera_make or not camera_model:
            return image.replace(metadata={"lens_correct": {"skipped": "no camera EXIF"}})

        import lensfunpy

        db = lensfunpy.Database()
        result = _find_camera_and_lens(
            db,
            camera_make,
            camera_model,
            exif.get("lens_model"),
            exif.get("lens_make"),
        )

        if result is None:
            return image.replace(
                metadata={
                    "lens_correct": {
                        "skipped": "camera/lens not in lensfun database",
                        "camera": camera_model,
                        "lens": exif.get("lens_model"),
                    }
                }
            )

        cam, lens = result
        focal_length = exif.get("focal_length", 0.0)
        aperture = exif.get("aperture", 0.0)
        distance = float(params.get("distance", 10.0))

        if not focal_length or focal_length <= 0 or not aperture or aperture <= 0:
            return image.replace(
                metadata={
                    "lens_correct": {
                        "skipped": "missing focal_length or aperture in EXIF",
                        "camera": cam.model,
                        "lens": lens.model,
                    }
                }
            )

        corrected, corrections = _apply_corrections(
            image.data,
            cam,
            lens,
            float(focal_length),
            float(aperture),
            distance,
            correct_distortion=bool(params.get("correct_distortion", True)),
            correct_vignetting=bool(params.get("correct_vignetting", True)),
            correct_tca=bool(params.get("correct_tca", True)),
        )

        return image.replace(
            data=corrected,
            metadata={
                "lens_correct": {
                    "camera": cam.model,
                    "lens": lens.model,
                    "crop_factor": cam.crop_factor,
                    "focal_length": float(focal_length),
                    "aperture": float(aperture),
                    "distance": distance,
                    "corrections": corrections,
                }
            },
        )
