# ImageData 数据结构与流水线数据流

## 核心数据结构

整个 pipeline 只有一个核心数据结构 `ImageData`，从加载到保存全程流转：

```python
class ImageData:
    data: np.ndarray       # (H, W, 3) float32 RGB 像素数据
    encoding: Encoding      # LINEAR 或 SRGB
    source_format: str      # "raw" / "jpg" / "heif" / "png" / "tiff"
    source_bit_depth: int   # 8 / 10 / 12 / 14 / 16
    original_path: Path     # 原始文件路径
    metadata: dict          # 累积的处理器元数据
```

### 字段说明

| 字段 | 用途 |
|------|------|
| `data` | 实际的像素数据，统一为 float32 三通道 RGB |
| `encoding` | 标记数据当前的编码状态——LINEAR（线性光）或 SRGB（gamma 编码） |
| `source_format` | 记录原始文件格式，用于 pipeline 中按格式跳过不适用的处理器 |
| `source_bit_depth` | 记录原始位深（RAW 常见 12/14/16，HEIF 8/10，JPG 8），用于保存时选择输出位深 |
| `metadata` | 每个处理器执行后往里追加自己的处理信息，贯穿整个 pipeline |

### 编码状态

```python
class Encoding(Enum):
    LINEAR = "linear"   # 线性光，值域 [0.0, 可能 > 1.0]
    SRGB = "srgb"       # sRGB gamma 编码，值域 [0.0, 1.0]
```

- **LINEAR**: 像素值与物理光照强度成正比。RAW 文件加载后的状态，高光区域值可能超过 1.0
- **SRGB**: 经过 sRGB gamma 曲线编码。人眼感知更均匀，所有 AI 模型期望的输入格式

### 不可变更新

处理器不直接修改 `ImageData`，而是通过 `replace()` 返回新实例：

```python
# 处理器内部
return image.replace(
    data=new_data,
    metadata={"white_balance": {"temperature": 5500}}
)
```

`replace()` 会自动将新 metadata 与已有的合并，确保每一步的处理痕迹都保留。

## sRGB 传递函数

按 IEC 61966-2-1 标准实现，不是简单的 gamma 2.2 近似：

- `linear_to_srgb()`: 低值段线性（`x * 12.92`）+ 高值段幂函数（`1.055 * x^(1/2.4) - 0.055`）
- `srgb_to_linear()`: 上述函数的逆运算

`ImageData` 提供便捷方法：
- `to_linear()` — sRGB → 线性光，已经是 LINEAR 则返回自身
- `to_srgb()` — 线性光 → sRGB 并裁切到 [0, 1]，已经是 SRGB 则返回自身

## 加载策略

不同格式加载后的状态不同：

| 格式 | 编码 | 加载方式 | 值域 | 位深 |
|------|------|----------|------|------|
| RAW (.cr2/.nef/.arw/...) | **LINEAR** | rawpy, `gamma=(1,1)`, Blend | [0, ~0.5-0.8] (Blend WB 归一化后) | 16-bit |
| HEIF (.heic/.heif) | SRGB | pillow-heif | [0, 1] | 8 或 10-bit |
| JPG (.jpg/.jpeg) | SRGB | Pillow | [0, 1] | 8-bit |
| PNG (.png) | SRGB | Pillow | [0, 1] | 8 或 16-bit |
| TIFF (.tiff/.tif) | SRGB | Pillow | [0, 1] | 8 或 16-bit |

关键点：
- RAW 用 `gamma=(1,1)` 输出线性数据，不烘焙 sRGB gamma，保留完整动态范围
- Blend 高光恢复模式内部除以 max(WB) 防止溢出，导致整体偏暗——由下游 auto-EV 自动补偿
- HEIF 检测 iPhone 10-bit 照片模式
- PNG/TIFF 支持 16-bit 灰度/RGB 加载

## 数据流

### RAW 路径

```
photo.cr2
  │
  ▼  rawpy: Bayer 分析 → Blend 高光恢复 → DHT demosaic → gamma=(1,1), 16-bit
  │
  ImageData(encoding=LINEAR, 值域 [0.0, ~0.5-0.8])
  │  线性光数据，Blend WB 归一化后整体偏暗
  │
  ▼  [5] lens_correct
  │  lensfunpy 镜头校正：暗角 → TCA + 畸变
  │  仍然 LINEAR（线性空间做暗角补偿更精确）
  │
  ▼  [10] raw_develop
  │  auto-EV 曝光补偿 → Hable filmic tone map → linear_to_srgb()
  │  ★ 变为 SRGB [0, 1] ★
  │
  ▼  [20+] 后续处理器（降噪、校正、调色等，全部在 SRGB 空间）
  │
  ▼  save_image → output.jpg
```

### JPG / HEIF / PNG 路径

```
photo.jpg
  │
  ▼  Pillow 加载, /255.0 归一化
  │
  ImageData(encoding=SRGB, 值域 [0.0, 1.0])
  │
  ▼  [5] lens_correct — 跳过 (supported_inputs=["raw"])
  ▼  [10] raw_develop — 跳过 (supported_inputs=["raw"])
  │
  ▼  [11] exposure_adjust
  │  轻微 S 曲线: 微弱提亮暗部、压制高光
  │  仍然 SRGB
  │
  ▼  [20+] 后续处理器（降噪、校正、调色等）
  │
  ▼  save_image → output.jpg
```

### 汇合点

**RAW 和非 RAW 路径通过互斥的处理器各自到达 SRGB：**
- RAW → `raw_develop`（order 10）：auto-EV + Hable filmic + linear_to_srgb()
- JPG/HEIF/PNG/TIFF → `exposure_adjust`（order 11）：轻微 S 曲线

从 order 20 开始，无论输入是什么格式，数据都统一为 SRGB [0, 1]，后续处理器在同一个标准空间工作。

## 为什么 RAW 要保持 LINEAR

RAW 文件的核心价值是**动态范围**。传感器记录的是线性光强度，12-14 bit 的数据中包含了大量高光和暗部细节。

如果加载时直接应用 sRGB gamma：
- 高光区域被裁切到 1.0，过曝部分数据永久丢失
- 暗部被 gamma 曲线压缩，量化精度降低

保持 LINEAR 编码到 `raw_develop` 阶段：
- 线性空间做曝光补偿，`data * 2` 就是 +1EV，物理上精确
- Hable filmic 曲线优雅地压缩动态范围，高光自然滚降而非硬裁切

## Pipeline 跳过逻辑

每个处理器执行前经过两道检查：

1. **格式匹配** — 处理器的 `supported_inputs` 与 `image.source_format` 对比。`["any"]` 匹配一切，`["raw"]` 只匹配 RAW 文件
2. **状态检查** — `NEEDS_MODEL`（模型未下载）或 `UNAVAILABLE`（缺少依赖）的处理器跳过

跳过的步骤记入 `PipelineResult.skipped`，执行的步骤记入 `steps_run`（含耗时和当前编码状态）。

## 保存

`save_image()` 有一个安全网：如果传入的 `ImageData` 仍然是 LINEAR（正常流程不应该出现，但防御性编程），会自动调用 `to_srgb()` 转换后再保存。

当前统一输出 8-bit RGB。16-bit TIFF/PNG 输出待后续实现。

