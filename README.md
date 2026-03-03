# Shotline

Automated photo processing pipeline. Supports RAW, JPG, HEIF — from camera to baseline edit with full control over every step.

## Install

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/holdjun/Shotline.git
cd Shotline
uv sync                 # core
uv sync --extra lens    # + lens correction
uv sync --extra ai      # + AI models
uv sync --extra all     # everything
```

## Usage

```bash
shotline run photo.arw                          # single file
shotline run ./photos/ -o ./output/             # directory
shotline run photo.jpg -s denoise,color_grade   # specific steps
shotline list                                   # show available steps
```

Config: `shotline.toml`, override with `shotline run photo.arw -c my_config.toml`.

## Processing Steps

| Step | Description |
|------|-------------|
| `lens_correct` | Lens distortion, vignetting, and chromatic aberration correction |
| `raw_develop` | Adaptive RAW decoding with auto-exposure and Hable filmic tone mapping |
| `exposure_adjust` | S-curve exposure refinement |
| `denoise` | Noise reduction |
| `horizon` | Horizon leveling |
| `white_balance` | White balance correction |
| `color_grade` | Color grading |
| `auto_crop` | Automatic crop with aspect ratio control |

## Docs

Technical documentation and architecture details live under [`docs/`](docs/).

## Claude Code Skills

- **`/setup`** — Configure GitHub repo settings
- **`/submit`** — Code-to-PR workflow (lint, test, review, push, create PR, monitor CI)

## License

MIT
