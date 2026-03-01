"""Auto-import processors to trigger @register_processor."""

from shotline.processors import (  # noqa: F401
    auto_crop,
    color_grade,
    content_remove,
    denoise,
    horizon,
    raw_develop,
    super_res,
    white_balance,
)
