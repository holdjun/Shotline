# Shotline Framework (Archived)

> **Note:** This is the initial planning document from project inception. Some details (directory structure, processor names, data flow) no longer reflect the current implementation. Kept for historical reference.

## Context

Shotline 是一个 Python CLI 图片处理工具，支持 RAW/HEIF/JPG 输入，通过可组合的处理步骤自动化处理照片。每个步骤可单独运行或组合为流水线。未来会被 Swift macOS app 调用。

当前仓库为空，本次实现框架骨架——处理器为 stub，跑通 CLI → Pipeline → Processor 完整链路。

## 目录结构

```
src/shotline/
├── __init__.py
├── cli.py                   # Typer CLI
├── pipeline.py              # 流水线编排（简单串联）
├── config.py                # 配置 + TOML 加载
├── image.py                 # ImageData 数据结构 + sRGB 传递函数
├── io.py                    # 图片 I/O (RAW/HEIF/JPG)
├── models.py                # AI 模型下载/缓存
├── processor.py             # BaseProcessor ABC + 注册表
└── processors/
    ├── __init__.py
    ├── raw_develop.py       # RAW 开发 (rawpy)
    ├── tone_map.py          # 色调映射 (filmic S 曲线)
    ├── denoise.py           # AI 去噪 (NAFNet)
    ├── horizon.py           # 水平矫正 (OpenCV)
    ├── white_balance.py     # 白平衡
    ├── color_grade.py       # 自动调色 (3DLUT)
    ├── auto_crop.py         # 自动裁图
    ├── super_res.py         # 超分辨率 (Real-ESRGAN)
    └── content_remove.py    # 内容感知移除 (未来)
tests/
├── conftest.py
├── test_processor.py        # 处理器注册/接口测试
├── test_pipeline.py         # Pipeline 测试
├── test_image.py            # ImageData + sRGB 传递函数测试
├── test_io.py               # I/O 测试
└── test_cli.py              # CLI 集成测试
```

## 核心设计

### BaseProcessor (`src/shotline/processor.py`)

- ABC：`meta()` 返回名称/描述/排序，`process(ImageData, params) -> ImageData`
- `@register_processor` 装饰器自动注册
- 图片用 `ImageData` 传递（见 [image-data-flow.md](image-data-flow.md)）
- `status()` 返回 available / needs_model / unavailable

### Pipeline (`src/shotline/pipeline.py`)

简单串联：接收 step 名称列表，按序执行。模型未下载或输入类型不匹配时跳过。`ImageData` 在处理器之间直接传递，metadata 自动累积。

### CLI (`src/shotline/cli.py`)

```bash
shotline run photo.cr2 -o output.jpg           # 默认步骤
shotline run photo.jpg -s denoise,color_grade   # 指定步骤
shotline run ./photos/ -o ./output/             # 批量处理
shotline run photo.cr2 --json                   # JSON 输出 (供 Swift 调用)
shotline list                                    # 列出处理器
shotline models download all                     # 下载模型
```

### 依赖 (`pyproject.toml`)

- **核心**: typer, rich, pydantic, numpy, Pillow, rawpy, pillow-heif
- **[ai]**: torch, torchvision, opencv-python-headless, ultralytics
- **[dev]**: pytest, pytest-cov, ruff, basedpyright

## 处理器技术选型

| 步骤 | 方案 | 模型大小 |
|------|------|----------|
| RAW 开发 | rawpy (LibRaw) | 无模型 |
| AI 去噪 | NAFNet width32 | ~17MB |
| 水平矫正 | OpenCV Hough Lines | 无模型 |
| 白平衡 | rawpy camera WB / Deep WB | ~25MB |
| 自动调色 | Image-Adaptive-3DLUT | ~30MB |
| 自动裁图 | YOLOv8 + 三分法 | ~6MB |
| 超分辨率 | Real-ESRGAN x4plus | ~64MB |
| 内容移除 | LaMa / IOPaint | ~100MB (未来) |

## 实现步骤

1. `pyproject.toml` + `.gitignore` + `src/shotline/__init__.py`
2. `processor.py` (BaseProcessor + registry)
3. `config.py` + `io.py` + `models.py`
4. `pipeline.py` + `cli.py`
5. `processors/` 下 8 个 stub 文件
6. `tests/` 完整测试
7. CI workflow + `CLAUDE.md` 更新

## 验证

```bash
uv sync
shotline list                    # 列出 9 个处理器
shotline run test.jpg -o out.jpg # stub pipeline 跑通
uv run pytest                    # 测试通过
uv run ruff check src/ tests/   # lint 通过
```
