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

默认 **Blend**（rawpy `highlight_mode=2`）。

rawpy 提供三种高光处理模式：
- **Clip**：过曝通道硬裁切到白电平，高光边缘易出现色彩偏移
- **Blend**：过曝像素平滑过渡到中性白，无色彩伪影
- **Rebuild**：从相邻像素重建过曝通道的颜色信息

选择 Blend：行业标准默认（Lightroom、Capture One、RawTherapee 均默认高光重建）。Blend 内部会做 WB 归一化（除以 max(WB)），输出整体偏暗，由下游 auto-EV 自动补偿。

用户可通过 `highlight_mode = "Clip"` 或 `"Rebuild"` 覆盖。

## 降噪策略

默认 **FBDD Off**（不做 pre-demosaic 降噪）。

rawpy 提供三级 FBDD pre-demosaic 降噪：
- **Off**：不降噪
- **Light**：轻度空间滤波
- **Full**：强空间滤波

选择 Off：pre-demosaic 空间滤波会破坏 Bayer 域通道间相关性，导致 demosaic 伪影和细节损失。降噪交给 pipeline 后续的 post-demosaic denoiser（NAFNet, order=20）。

用户可通过 `fbdd_noise_reduction = "Light"` 或 `"Full"` 覆盖。

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

## 参数体系

从 `[processor_params.raw_develop]` 读取，分三层：

**日常参数**（用户可能按需调整）：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ev` | float | `0.0` | 手动曝光补偿，叠加在 auto-EV 之上 |
| `auto_expose` | bool | `true` | 自动曝光；关闭则纯手动 ev |
| `demosaic_algorithm` | str | `"DHT"` | Bayer 域插值算法 |
| `highlight_mode` | str | `"Blend"` | 高光恢复: "Blend"/"Clip"/"Rebuild" |
| `fbdd_noise_reduction` | str | `"Off"` | Pre-demosaic 降噪: "Off"/"Light"/"Full" |

**风格参数**（定一次基本不动）：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `bright` | float | `1.0` | 亮度乘数（与 ev 功能重叠，一般不用） |
| `white_point` | float | `11.2` | Hable filmic 白点（Uncharted 2 经典默认） |
| `hable_A`..`hable_F` | float | 见上方 Hable 章节 | Hable 曲线形状参数，极少需要逐个调 |

**rawpy 透传参数**（基本不需要调整）：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `exp_preserve_highlights` | float | `0.75` | 高光保护（下游 Hable 已处理） |
| `chromatic_aberration` | list/null | `null` | 色差校正（lensfun 已处理 TCA） |
| `noise_thr` | float/null | `null` | 小波降噪（降噪交给 NAFNet） |
| `median_filter_passes` | int | `0` | 中值滤波轮次 |

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
