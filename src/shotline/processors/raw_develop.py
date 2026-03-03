"""RAW development: auto-exposure → Hable filmic tone map → sRGB output."""

from __future__ import annotations

from typing import Any

import numpy as np

from shotline.image import Encoding, ImageData, linear_to_srgb
from shotline.processor import BaseProcessor, ProcessorMeta, register_processor

# ── Hable (Uncharted 2) filmic operator ──
# Parameter names follow John Hable's GDC 2010 notation (A-F, W).

HABLE_DEFAULTS = {
    "A": 0.15,  # shoulder strength
    "B": 0.50,  # linear strength
    "C": 0.10,  # linear angle
    "D": 0.20,  # toe strength
    "E": 0.02,  # toe numerator
    "F": 0.30,  # toe denominator
}
HABLE_DEFAULT_W = 11.2  # white point


def _hable_operator(x: np.ndarray, p: dict[str, float]) -> np.ndarray:
    """Hable/Uncharted 2 tone mapping operator.

    f(x) = ((x*(A*x + C*B) + D*E) / (x*(A*x + B) + D*F)) - E/F
    """
    a, b, c, d, e, f = p["A"], p["B"], p["C"], p["D"], p["E"], p["F"]
    return ((x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f)) - e / f


def _hable_filmic(data: np.ndarray, white_point: float = 11.2, **params: float) -> np.ndarray:
    """Apply Hable filmic tone mapping with white point normalization.

    Maps linear HDR [0, inf) → [0, 1] with natural toe and shoulder.
    """
    p = {k: params.get(k, v) for k, v in HABLE_DEFAULTS.items()}
    mapped = _hable_operator(np.maximum(data, 0.0), p)
    white_mapped = _hable_operator(np.array([white_point], dtype=np.float32), p)
    white_scale = 1.0 / float(white_mapped[0])
    return np.clip(mapped * white_scale, 0.0, 1.0).astype(np.float32)


def _compute_auto_ev(data: np.ndarray) -> tuple[float, dict[str, float]]:
    """Compute auto-exposure EV using log-average scene key.

    Uses geometric mean of luminance (Rec.709) with percentile clipping
    to target 18% gray (0.18 in linear light).

    Returns (auto_ev, stats_dict).
    """
    luminance = 0.2126 * data[..., 0] + 0.7152 * data[..., 1] + 0.0722 * data[..., 2]

    p1, p99 = np.percentile(luminance, [1, 99])
    clipped = luminance[(luminance >= p1) & (luminance <= p99)]

    if clipped.size == 0:
        return 0.0, {"scene_key": 0.0, "p1": float(p1), "p99": float(p99)}

    eps = 1e-6
    scene_key = float(np.exp(np.mean(np.log(clipped + eps))))

    auto_ev = 0.0 if scene_key < eps else float(np.log2(0.18 / scene_key))
    auto_ev = max(-4.0, min(4.0, auto_ev))

    return auto_ev, {
        "scene_key": scene_key,
        "p1": float(p1),
        "p99": float(p99),
    }


@register_processor
class RawDevelopProcessor(BaseProcessor):
    def meta(self) -> ProcessorMeta:
        return ProcessorMeta(
            name="raw_develop",
            display_name="RAW Development",
            description="Full RAW pipeline: auto-exposure, tone map, sRGB output",
            order=10,
            supported_inputs=["raw"],
        )

    def process(self, image: ImageData, params: dict[str, Any] | None = None) -> ImageData:
        params = params or {}
        auto_expose = bool(params.get("auto_expose", True))
        ev = float(params.get("ev", 0.0))
        bright = float(params.get("bright", 1.0))

        data = image.data
        loader_meta = image.metadata.get("raw_loader", {})

        # ── 1. Auto-exposure: log-average scene key → 18% gray ──
        auto_ev = 0.0
        auto_ev_stats: dict[str, float] = {}
        if auto_expose:
            auto_ev, auto_ev_stats = _compute_auto_ev(data)
            if auto_ev != 0.0:
                data = data * (2.0**auto_ev)

        # ── 2. Manual EV (skip if already applied at load via exp_shift) ──
        if ev != 0.0 and loader_meta.get("exp_shift_applied", 1.0) == 1.0:
            data = data * (2.0**ev)

        # ── 3. Brightness scaling ──
        if bright != 1.0:
            data = data * bright

        # Ensure non-negative before tone mapping
        data = np.maximum(data, 0.0).astype(np.float32)

        # ── 4. Hable filmic tone mapping: LINEAR → [0, 1] ──
        wp = float(params.get("white_point", HABLE_DEFAULT_W))
        hable_params = {
            k: float(params.get(f"hable_{k}", HABLE_DEFAULTS[k])) for k in HABLE_DEFAULTS
        }
        mapped = _hable_filmic(data, white_point=wp, **hable_params)

        # ── 5. Linear → sRGB gamma ──
        srgb = np.clip(linear_to_srgb(mapped), 0.0, 1.0).astype(np.float32)

        return image.replace(
            data=srgb,
            encoding=Encoding.SRGB,
            metadata={
                "raw_develop": {
                    "auto_expose": auto_expose,
                    "auto_ev": auto_ev,
                    "auto_ev_stats": auto_ev_stats,
                    "ev": ev,
                    "bright": bright,
                    "exp_shift_applied_at_load": loader_meta.get("exp_shift_applied", 1.0) != 1.0,
                    "tone_map": {
                        "method": "hable_filmic",
                        "white_point": wp,
                        **{f"hable_{k}": v for k, v in hable_params.items()},
                    },
                }
            },
        )
