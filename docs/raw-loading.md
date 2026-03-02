# RAW 加载与自适应处理流水线

## 概述

RAW 处理由两个阶段完成：`_load_raw`（信号级解码 + 自适应选择）→ `raw_develop`（自动曝光 + Hable filmic + sRGB 输出）。

JPG/HEIF 走独立路径：`exposure_adjust`（轻微 S-curve）。

```
文件(*.ARW/.CR2/.DNG)
    │
    ▼
_load_raw()
    ├─ _analyze_bayer() — 分析饱和率、噪声级别
    ├─ 自适应选择: 有过曝→Blend, 无过曝→Clip; 高噪声→FBDD Full
    ├─ rawpy.postprocess()
    └─ 若 Blend: WB 补偿恢复亮度，保留 HDR 高光 (>1.0)
    │
    ▼
ImageData(LINEAR, [0, ~1.5-2.0])
    │
    ▼
raw_develop
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

## 自适应 Bayer 分析

`_analyze_bayer(raw)` 在 `postprocess()` 之前分析 `raw.raw_image_visible`：

```python
normalized = (bayer - black_level) / (white_level - black_level)
saturation_ratio = (normalized >= 0.95).mean()
noise_std = normalized[normalized < 0.05].std()
```

| 指标 | 默认阈值 | 决策 | 配置键 |
|------|----------|------|--------|
| 饱和率 >= threshold | `0.005` | Blend + WB 补偿 | `saturation_threshold` |
| 饱和率 < threshold | `0.005` | Clip | `saturation_threshold` |
| 噪声 < low | `0.003` | FBDD Off | `noise_low` |
| low <= 噪声 < high | `0.003-0.008` | FBDD Light | `noise_low`, `noise_high` |
| 噪声 >= high | `0.008` | FBDD Full | `noise_high` |

所有阈值均可通过 `[processor_params.raw_develop]` 配置，省略时使用默认值。用户也可直接设置 `highlight_mode` / `fbdd_noise_reduction` 覆盖自适应选择。

## WB 补偿原理

Blend 高光恢复将所有通道除以 `max(WB multiplier)` 防止溢出，导致整图变暗约 `max(WB)/min(WB)` 倍（Sony 约 2.1-2.8x）。

`_load_raw` 在 Blend 解码后乘以补偿系数：

```python
wb_compensation = max(camera_wb[:3]) / min(camera_wb[:3])
data *= wb_compensation  # 恢复中间调，高光保留 >1.0 HDR 数据
```

Clip 模式不需要补偿（`wb_compensation = 1.0`）。

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
| `highlight_mode` | str/null | `null`（自适应） | 省略=自适应, 设置=强制 |
| `fbdd_noise_reduction` | str/null | `null`（自适应） | 省略=自适应, 设置=强制 |
| `saturation_threshold` | float | `0.005` | Bayer 饱和率阈值 |
| `noise_low` | float | `0.003` | FBDD Off/Light 分界 |
| `noise_high` | float | `0.008` | FBDD Light/Full 分界 |
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
    "bayer_analysis": {
        "saturation_ratio": 0.018,
        "noise_std": 0.002,
        "recommended_highlight": "Blend",
        "recommended_fbdd": "Off"
    },
    "wb_compensation": 2.1,
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
# 自定义阈值
saturation_threshold = 0.01
noise_high = 0.01
# 自定义 Hable
white_point = 8.0
hable_D = 0.25  # 增大 toe 提升暗部
```
