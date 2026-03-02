# RAW 加载与开发

## 概述

RAW 加载是 Shotline 处理流程的入口，负责将相机传感器原始数据解码为 `ImageData`（LINEAR float32）。

架构位置：

```
文件(*.ARW/.CR2/.DNG)
    │
    ▼
_load_raw()          ← 信号级处理（demosaic、高光恢复、WB、降噪）
    │
    ▼
ImageData(LINEAR)
    │
    ▼
raw_develop          ← 创意调整（亮度、EV fallback）
    │
    ▼
tone_map             ← 动态范围压缩（LINEAR → SRGB）
```

`_load_raw` 处理所有必须在 Bayer 域或解码时完成的操作；`raw_develop` 处理线性空间后处理；`tone_map` 负责色调映射。

## 架构边界

| 关注点 | 归属 | 理由 |
|--------|------|------|
| Demosaic 算法 | `_load_raw` | Bayer 域信号重建，必须在解码时完成 |
| 高光恢复模式 | `_load_raw` | 传感器饱和数据恢复，解码后无法补救 |
| FBDD 降噪 | `_load_raw` | 在 Bayer 域执行（demosaic 之前），效果远好于后期 |
| 色差校正 | `_load_raw` | 光学校正，在 Bayer 域执行 |
| 白平衡选择 | `_load_raw` | rawpy 在解码时烘焙 WB 矩阵 |
| 元数据提取 | `_load_raw` | raw handle 只在加载时可用 |
| EV 曝光补偿 | `_load_raw`（`exp_shift`） | rawpy 的 exp_shift 在像素级执行，能配合高光保护 |
| 亮度缩放 | `raw_develop`（`bright`） | 线性空间后处理调整 |
| 色调映射 | `tone_map` | LINEAR → SRGB 的动态范围压缩 |

## `_load_raw` 参数体系

所有参数从 `[processor_params.raw_develop]` 读取，通过 pipeline 传递给 `load_image(path, raw_params=...)`。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `demosaic_algorithm` | str | `"DHT"` | Bayer 域插值算法，DHT 质量优于 AHD 且无需 GPL 包 |
| `highlight_mode` | str/int | `"Blend"` | 过曝通道恢复策略，Blend 从未裁切通道重建 |
| `fbdd_noise_reduction` | str | `"Light"` | Bayer 域预降噪，在 demosaic 之前执行 |
| `ev` | float | `0.0` | 曝光补偿（档位），转为 `exp_shift = 2^ev` |
| `exp_preserve_highlights` | float | `0.75` | 曝光补偿时的高光保护强度 (0-1) |
| `chromatic_aberration` | list/null | `null` | 色差校正 `[red_scale, blue_scale]` |
| `noise_thr` | float/null | `null` | 后 demosaic 小波降噪阈值 |
| `median_filter_passes` | int | `0` | 后 demosaic 中值滤波轮次 |
| `bright` | float | `1.0` | 亮度乘数（在 `raw_develop` 处理器中执行） |

### 默认值选择理由

**DHT > AHD**：DHT（Demosaicing by High-frequency Texture）在边缘和纹理区域的还原质量优于 AHD，且不需要额外的 GPL 许可包。如果安装了 GPL2 demosaic pack，可配置为 LMMSE 获得更好的高 ISO 噪声抑制。当请求的算法不可用时（如 LMMSE 缺少 GPL2 包），自动回退到 DHT。

**Blend > Clip**：默认 Clip 直接丢弃过曝通道数据。Blend 从未裁切的通道重建高光信息，在多数场景下表现稳定。ReconstructDefault 在某些高对比度边缘可能产生伪影，Blend 更保守。

**Light FBDD**：Bayer 域降噪在 demosaic 之前执行，能抑制热像素和随机噪声，避免这些噪声被 demosaic 算法放大为色彩伪影。Light 模式平衡噪声抑制和细节保留。

## 字符串→枚举映射

TOML 中使用可读字符串，`_load_raw` 内部映射为 rawpy 枚举值：

```python
# Demosaic: "AHD", "DHT", "AAHD", "DCB", "PPG", "VNG" (always available)
#           "LMMSE", "MODIFIED_AHD", "VCD" (need GPL2 pack)
#           "AMAZE" (needs GPL3 pack)
rawpy.DemosaicAlgorithm.DHT

# Highlight: "Clip", "Blend", "ReconstructDefault", 或整数 3-9
rawpy.HighlightMode.Blend

# FBDD: "Off", "Light", "Full"
rawpy.FBDDNoiseReductionMode.Light
```

`highlight_mode` 支持整数值 3-9，对应 libraw 的不同高光重建级别（越高重建越激进）。

## 白平衡 Fallback 策略

```
读取 raw.camera_whitebalance
    │
    ├── 任何通道 != 0 → use_camera_wb=True
    │
    └── 全部为 0（元数据缺失）→ use_auto_wb=True
```

rawpy 在解码时将 WB 矩阵烘焙到像素值中，因此 WB 选择必须在 `_load_raw` 完成。`camera_whitebalance` 来自拍摄时相机记录的色温数据，是最准确的。当这个数据缺失时（某些老旧或非标准 RAW 格式），自动切换到 rawpy 的自动白平衡算法。

## 元数据字段

`_load_raw` 在 `ImageData.metadata["raw_loader"]` 中记录完整的加载参数和传感器信息：

```python
{
    "camera_whitebalance": [2.1, 1.0, 1.5, 1.0],  # RGBG 通道白平衡乘数
    "daylight_whitebalance": [2.0, 1.0, 1.4, 1.0],  # 日光下的参考白平衡
    "black_level_per_channel": [512, 512, 512, 512],  # 每通道黑电平
    "white_level": 16383,              # 传感器饱和度（14-bit = 16383）
    "color_desc": "RGBG",              # 滤色器排列
    "num_colors": 3,                   # 输出颜色通道数
    "sizes": {"width": 6000, "height": 4000},  # 传感器尺寸
    "demosaic_algorithm": "LMMSE",     # 实际使用的 demosaic 算法
    "highlight_mode": "Blend",         # 实际使用的高光模式
    "fbdd_noise_reduction": "Light",   # 实际使用的 FBDD 模式
    "exp_shift_applied": 2.0,          # 实际应用的曝光倍数（1.0=未应用）
    "exp_preserve_highlights": 0.75,   # 高光保护强度
    "used_camera_wb": true,            # 是否使用了相机 WB
    "used_auto_wb": false,             # 是否使用了自动 WB
}
```

下游处理器可通过 `image.metadata["raw_loader"]` 读取这些信息。例如 `raw_develop` 检查 `exp_shift_applied` 决定是否需要 fallback EV 乘法。

## EV 补偿流程

EV（曝光值）补偿的最佳执行位置是 rawpy 的 `exp_shift` 参数，因为它：
1. 在像素级别执行，早于 output scaling
2. 能配合 `exp_preserve_highlights` 保护高光区域
3. 不会产生简单乘法导致的高光截断

流程：

```
用户配置 ev = 1.0（增加 1 档曝光）
    │
    ▼
pipeline.run()
    │ 读取 config.get_processor_params("raw_develop")
    │ 传递 raw_params 给 load_image()
    │
    ▼
_load_raw(path, raw_params={"ev": 1.0, ...})
    │ exp_shift = 2^1.0 = 2.0
    │ rawpy.postprocess(exp_shift=2.0, exp_preserve_highlights=0.75)
    │ metadata["raw_loader"]["exp_shift_applied"] = 2.0
    │
    ▼
raw_develop.process(image, params={"ev": 1.0})
    │ 检查 metadata["raw_loader"]["exp_shift_applied"] == 2.0
    │ 已在加载时应用 → 跳过 EV 乘法
    │
    ▼
ImageData（EV 已在最优位置应用）
```

当图片不是 RAW 格式时（如 JPG），`_load_raw` 不会被调用，metadata 中没有 `raw_loader`。此时 `raw_develop` 处理器也不会执行（因为 `supported_inputs=["raw"]`）。

## 配置示例

```toml
[processor_params.raw_develop]
ev = 0.5                            # 增加半档曝光
exp_preserve_highlights = 0.75      # 高光保护
demosaic_algorithm = "DHT"          # 或 AHD, AAHD, DCB, PPG, VNG, LMMSE (需 GPL2)
highlight_mode = "Blend"            # 或 Clip, ReconstructDefault, 3-9
fbdd_noise_reduction = "Light"      # 或 Off, Full
# chromatic_aberration = [1.0, 1.0] # 色差校正
# noise_thr = 200.0                 # 小波降噪
# median_filter_passes = 0          # 中值滤波
bright = 1.2                        # 亮度增加 20%
```
