"""AI model download and cache management."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from shotline.config import load_config


@dataclass
class ModelSpec:
    id: str
    display_name: str
    url: str
    filename: str
    size_mb: float
    sha256: str | None = None


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "nafnet_denoise": ModelSpec(
        id="nafnet_denoise",
        display_name="NAFNet Denoising",
        url="",  # TBD
        filename="nafnet_denoise.pth",
        size_mb=17.0,
    ),
    "deep_wb": ModelSpec(
        id="deep_wb",
        display_name="Deep White Balance",
        url="",
        filename="deep_wb.pth",
        size_mb=25.0,
    ),
    "3dlut_color": ModelSpec(
        id="3dlut_color",
        display_name="Image-Adaptive 3D LUT",
        url="",
        filename="3dlut_color.pth",
        size_mb=30.0,
    ),
    "yolov8_crop": ModelSpec(
        id="yolov8_crop",
        display_name="YOLOv8 Subject Detection",
        url="",
        filename="yolov8n.pt",
        size_mb=6.0,
    ),
    "real_esrgan": ModelSpec(
        id="real_esrgan",
        display_name="Real-ESRGAN Super Resolution",
        url="",
        filename="real_esrgan_x4.pth",
        size_mb=64.0,
    ),
}


class ModelManager:
    def __init__(self, cache_dir: Path | None = None):
        config = load_config()
        self.cache_dir = cache_dir or config.models.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def model_path(self, model_id: str) -> Path:
        spec = MODEL_REGISTRY[model_id]
        return self.cache_dir / spec.filename

    def is_downloaded(self, model_id: str) -> bool:
        return self.model_path(model_id).exists()

    def download(self, model_id: str) -> Path:
        if model_id == "all":
            for mid in MODEL_REGISTRY:
                self.download(mid)
            return self.cache_dir

        if self.is_downloaded(model_id):
            return self.model_path(model_id)

        spec = MODEL_REGISTRY[model_id]
        if not spec.url:
            raise NotImplementedError(f"Download URL not configured for model '{model_id}'")

        dest = self.model_path(model_id)
        import urllib.request

        urllib.request.urlretrieve(spec.url, dest)
        return dest

    def print_status(self) -> None:
        from rich.console import Console

        console = Console()
        for mid, spec in MODEL_REGISTRY.items():
            downloaded = self.is_downloaded(mid)
            icon = "[green]downloaded[/green]" if downloaded else "[yellow]not downloaded[/yellow]"
            console.print(f"  {spec.display_name:<35} ({spec.size_mb:.0f} MB) {icon}")

    def clean(self, model_id: str | None = None) -> None:
        if model_id:
            path = self.model_path(model_id)
            if path.exists():
                path.unlink()
        else:
            for mid in MODEL_REGISTRY:
                path = self.model_path(mid)
                if path.exists():
                    path.unlink()
