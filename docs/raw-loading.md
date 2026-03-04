# RAW 加载与处理流水线

## 概述

RAW 处理由三个阶段完成：`_load_raw`（信号级解码 + EXIF 提取）→ `lens_correct`（镜头光学校正）→ `raw_develop`（自动曝光 + Hable filmic + sRGB 输出）。

JPG/HEIF 走独立路径：`exposure_adjust`（轻微 S-curve）。

```
文件(*.ARW/.CR2/.DNG)
    │
    ▼
_load_raw()
    ├─ _analyze_bayer() — 计算饱和率（诊断 metadata）
    ├─ rawpy.postprocess() — Blend 高光恢复, FBDD Off, DHT demosaic
    └─ _extract_exif() — 提取相机/镜头 EXIF (make, model, focal, aperture)
    │
    ▼
ImageData(LINEAR, metadata.raw_loader.exif)
    │
    ▼
lens_correct (order=5, 线性空间)
    ├─ lensfun 数据库查询: camera_make/model → lens profile
    ├─ apply_color_modification() — 暗角校正 (像素亮度修正)
    └─ apply_subpixel_geometry_distortion() → cv2.remap() — TCA + 畸变校正
    │
    ▼
ImageData(LINEAR, 光学校正后)
    │
    ▼
raw_develop (order=10)
    ├─ _compute_auto_ev() — log-average scene key → 18% gray
    ├─ 应用 auto_ev + 手动 ev + bright
    ├─ _hable_filmic() — toe 提升暗部 + shoulder 压缩高光
    └─ linear_to_srgb()
    │
    ▼
ImageData(SRGB, [0, 1])  ← 可直接查看的图片
```

JPG/HEIF 路径：

```
文件(*.jpg/*.heic/*.hif)
    │
    ▼
load_image() → ImageData(SRGB)
    │
    ▼
exposure_adjust → 轻微 S-curve
    │
    ▼
ImageData(SRGB, [0, 1])
```

## 高光恢复策略

固定使用 **Blend** 模式（rawpy `highlight_mode=2`）。

**为什么不用 Clip**：Clip 模式在任一通道过曝时硬裁切到白电平，其他通道保持原值，导致高光边缘出现色彩偏移（如红花变青）。所有商业 RAW 处理器（Adobe Lightroom、Capture One、RawTherapee）默认执行高光重建，无一默认硬裁切。

**Blend 的工作原理**：将所有通道的 WB 乘子除以 `max(WB)`，确保 WB 应用后不会溢出。过曝像素平滑过渡到中性白，无色彩伪影。对未过曝图片，Blend 的输出与 Clip 数学等价（WB 归一化是纯线性缩放，不改变通道比例）。

**亮度影响**：Blend 的 WB 归一化会降低整体亮度（约为 `max(WB)/min(WB)` 倍，Sony 约 2-3x）。这由下游 `_compute_auto_ev()` 自动吸收——auto-EV 基于实际亮度计算曝光补偿，Blend 模式的欠曝自动被校正。

用户可通过 `highlight_mode = "Clip"` 或 `highlight_mode = "Rebuild"` 覆盖默认值。

## 降噪策略

固定使用 **FBDD Off**（不做 pre-demosaic 降噪）。

**原因**：
- 2024 学术共识（Inverse Problems and Imaging）：中低噪声应先 demosaic 后 denoise
- Pre-demosaic 空间滤波破坏 Bayer 域的通道间相关性，导致 demosaic 产生棋盘伪影和细节损失
- 所有商业 RAW 处理器（Adobe、darktable、Capture One）默认不做 pre-demosaic 空间降噪
- Pipeline 已有 post-demosaic NAFNet denoiser (order=20)，FBDD 冗余

用户可通过 `fbdd_noise_reduction = "Light"` 或 `"Full"` 覆盖。

## Bayer 分析

`_analyze_bayer(raw)` 在 `postprocess()` 之前分析 `raw.raw_image_visible`，计算 `saturation_ratio`（传感器接近饱和的像素比例）作为诊断 metadata：

```python
normalized = (bayer - black_level) / (white_level - black_level)
saturation_ratio = (normalized >= 0.95).mean()
```

## EXIF 提取

`_extract_exif(path, raw)` 从 RAW 文件提取相机/镜头 EXIF 信息：

- **主路径**：`exifread.process_file()` 直接解析 RAW 的 TIFF/EXIF header
- **Fallback**：`rawpy.extract_thumb()` → PIL `Image.getexif()` 从内嵌 JPEG 缩略图提取
- 提取失败返回空 dict，lens_correct 跳过

提取字段：`camera_make`, `camera_model`, `lens_make`, `lens_model`, `focal_length`, `aperture`, `iso`

存储于 `metadata["raw_loader"]["exif"]`。

## 镜头光学校正（lens_correct）

详见 [lens-correction.md](lens-correction.md)

## 自动曝光算法

`_compute_auto_ev(data)` 使用 log-average scene key：

```python
L = 0.2126*R + 0.7152*G + 0.0722*B      # Rec.709 luminance
p1, p99 = percentile(L, [1, 99])          # 抗极端值
scene_key = exp(mean(log(L_clipped + ε))) # 几何平均
auto_ev = log2(0.18 / scene_key)          # 校正到 18% gray
auto_ev = clamp(auto_ev, -4, +4)
```

- **18% gray (0.18)**：摄影测光标准中间调
- **几何平均**：对高光/阴影影响均衡
- **±4 档限制**：防止极端场景产生不合理校正

自动曝光后，手动 `ev` 叠加其上。`auto_expose=false` 退化为纯手动模式。

## Hable Filmic 色调映射

```python
f(x) = ((x*(A*x + C*B) + D*E) / (x*(A*x + B) + D*F)) - E/F
```

| 参数 | 默认值 | 作用 | 配置键 |
|------|--------|------|--------|
| A | 0.15 | Shoulder 强度 | `hable_A` |
| B | 0.50 | 线性段强度 | `hable_B` |
| C | 0.10 | 线性段角度 | `hable_C` |
| D | 0.20 | Toe 强度 | `hable_D` |
| E | 0.02 | Toe 分子 | `hable_E` |
| F | 0.30 | Toe 分母 | `hable_F` |
| W | 11.2 | 白点 | `white_point` |

## `_load_raw` 参数体系

从 `[processor_params.raw_develop]` 读取：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `demosaic_algorithm` | str | `"DHT"` | Bayer 域插值算法 |
| `highlight_mode` | str | `"Blend"` | 高光恢复: "Blend"/"Clip"/"Rebuild" |
| `fbdd_noise_reduction` | str | `"Off"` | Pre-demosaic 降噪: "Off"/"Light"/"Full" |
| `ev` | float | `0.0` | 曝光补偿（exp_shift） |
| `exp_preserve_highlights` | float | `0.75` | 高光保护 (0-1) |
| `chromatic_aberration` | list/null | `null` | 色差校正 |
| `auto_expose` | bool | `true` | 自动曝光 |
| `bright` | float | `1.0` | 亮度乘数 |
| `white_point` | float | `11.2` | Hable 白点 |
| `hable_A`..`hable_F` | float | 见上表 | Hable 参数 |

## 元数据

`raw_develop` 输出 `ImageData.metadata["raw_develop"]`：

```python
{
    "auto_expose": True,
    "auto_ev": 1.5,
    "auto_ev_stats": {"scene_key": 0.053, "p1": 0.01, "p99": 0.45},
    "ev": 0.0,
    "bright": 1.0,
    "exp_shift_applied_at_load": False,
    "tone_map": {
        "method": "hable_filmic",
        "white_point": 11.2,
        "hable_A": 0.15, ...
    }
}
```

`_load_raw` 输出 `metadata["raw_loader"]`：

```python
{
    "camera_whitebalance": [2.1, 1.0, 1.5, 1.0],
    "bayer_analysis": {"saturation_ratio": 0.018},
    "highlight_mode": "Blend",
    "fbdd_noise_reduction": "Off",
    "demosaic_algorithm": "DHT",
    ...
}
```

## 配置示例

```toml
[processor_params.raw_develop]
demosaic_algorithm = "DHT"
auto_expose = true
ev = 0.5
bright = 1.2
# 强制 Clip 高光（覆盖默认 Blend）
# highlight_mode = "Clip"
# 自定义 Hable
white_point = 8.0
hable_D = 0.25  # 增大 toe 提升暗部
```
