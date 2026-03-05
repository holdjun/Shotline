# 核心数据结构

Pipeline 内部的数据载体和运行机制。

> 色彩科学概念（Encoding、ColorSpace、传递函数）见 [color-science-primer.md](color-science-primer.md)。
> 处理流程和各阶段色彩空间见 [image-data-flow.md](image-data-flow.md)。

## ImageData

整个 pipeline 的唯一数据载体，从加载到保存全程流转。

```python
# src/shotline/image.py

class ImageData:
    data: np.ndarray       # (H, W, 3) float32 RGB 像素数据
    encoding: Encoding      # LINEAR 或 SRGB
    source_format: str      # "raw" / "jpg" / "heif" / "png" / "tiff"
    source_bit_depth: int   # 8 / 10 / 12 / 14 / 16
    original_path: Path     # 原始文件路径
    metadata: dict          # 累积的处理器元数据
```

| 字段 | 用途 |
|------|------|
| `data` | float32 三通道 RGB 像素数据 |
| `encoding` | 当前编码状态——`LINEAR`（线性光，值域可 >1.0）或 `SRGB`（gamma 编码，[0, 1]） |
| `source_format` | 原始文件格式，用于按格式跳过不适用的处理器 |
| `source_bit_depth` | 原始位深（RAW 12/14/16，HEIF 8/10，JPG 8） |
| `metadata` | 每个处理器追加自己的处理信息，贯穿整个 pipeline |

### 不可变更新

处理器不直接修改 `ImageData`，通过 `replace()` 返回新实例：

```python
return image.replace(
    data=new_data,
    metadata={"white_balance": {"temperature": 5500}}
)
```

`replace()` 自动将新 metadata 与已有的浅合并，确保每步处理痕迹都保留。

### 编码转换便捷方法

- `to_linear()` — sRGB → 线性光，已是 LINEAR 则返回自身
- `to_srgb()` — 线性光 → sRGB 并裁切到 [0, 1]，已是 SRGB 则返回自身

底层函数 `linear_to_srgb()` / `srgb_to_linear()` 按 IEC 61966-2-1 实现（分段函数，非简单 gamma 2.2 近似，详见 [primer §1](color-science-primer.md#1-色彩空间--编码两个正交维度)）。

## Pipeline 运行机制

```python
# src/shotline/pipeline.py → Pipeline.run()
```

处理器按 `order` 排序依次执行。每个处理器执行前经过两道检查：

1. **格式匹配** — `supported_inputs` 与 `image.source_format` 对比。`["any"]` 匹配一切，`["raw"]` 只匹配 RAW
2. **状态检查** — `NEEDS_MODEL`（模型未下载）或 `UNAVAILABLE`（缺少依赖）的处理器跳过

执行结果记录在 `PipelineResult`：
- `steps_run: list[dict]` — 每步名称、耗时、编码状态
- `skipped: list[str]` — 跳过的处理器名称
