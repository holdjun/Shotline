# 图像处理数据流

RAW 文件经过信号级解码、光学校正、曝光与色调映射输出 sRGB；JPEG/HEIF 经过轻度曝光调整后进入同一后处理流水线。

> 色彩科学基础（Encoding vs ColorSpace、线性域运算、tone mapping、色域转换）见 [color-science-primer.md](color-science-primer.md)。

## 行业对照

```
Lightroom Pipeline                     Shotline Pipeline
─────────────────                     ─────────────────
1. RAW Decode                         ┐
2. Black Level Subtraction            │
3. Linearization                      │  rawpy.postprocess()
4. White Balance                      │  一次调用完成
5. Demosaic (+ highlight recovery)    │  输出 linear ProPhoto RGB
6. Camera Profile / Color Matrix      ┘
   ─── 线性 ProPhoto RGB ───
7. Lens Correction (optical)           lens_correct (order=5)
8. Exposure Compensation               raw_develop: auto_ev + ev
   ─── 线性域 → 感知域 分界 ───
9.  Tone Curve                         raw_develop: hable_filmic
10. Color Space Conversion             ProPhoto → sRGB 色域映射
11. Transfer Function                  linear → sRGB 传递函数
   ─── sRGB ───
12. White Balance (残差修正)           white_balance (order=40)
13. Color Grading                      color_grade (order=50)
14. Sharpening + NR                    denoise (order=20)
15. Output                             save_image
```

## RAW 路径

```
photo.arw / .cr2 / .dng
  │
  ▼  io._load_raw()
  │
  │  rawpy.postprocess() 内部完成：
  │    ① RAW Decode — 厂商格式解包
  │    ② Black Level Subtraction — 减去传感器黑电平
  │    ③ Linearization — 非线性传感器响应校正为线性
  │    ④ White Balance — camera WB 通道增益（线性域乘法）
  │    ⑤ Demosaic — DHT 插值 + Blend 高光恢复
  │    ⑥ Color Matrix — sensor RGB → ProPhoto RGB
  │
  │  输出 float32 (H,W,3)，值域 [0, <1.0]（Blend 除以 max(WB) 导致偏暗）
  │
  ▼  ImageData(LINEAR, ProPhoto RGB)
  │
  │  ┌─────────────────────────────────────────────────┐
  │  │  线性 ProPhoto RGB 空间                           │
  │  │  宽色域保护饱和色不被裁切，线性域保证物理精确     │
  │  └─────────────────────────────────────────────────┘
  │
  ▼  [order=5] lens_correct ✅
  │  processors/lens_correct.py → LensCorrectProcessor.process()
  │
  │  _apply_corrections():
  │    1. 暗角校正 — lensfun apply_color_modification()，像素亮度乘法
  │    2. TCA + 畸变 — lensfun apply_subpixel_geometry_distortion() + cv2.remap()
  │    3. 黑边裁切 — _auto_crop_black_border()
  │
  ▼  ImageData(LINEAR, ProPhoto RGB, 光学校正后)
  │
  ▼  [order=10] raw_develop 🔧 需未来确认
  │  processors/raw_develop.py → RawDevelopProcessor.process()
  │
  │  Step 1: 自动曝光
  │    _compute_auto_ev() — Reinhard scene key → 18% gray
  │    data *= 2^auto_ev
  │
  │  Step 2: 手动 EV 偏移
  │    data *= 2^ev（叠加在 auto_ev 之上）
  │
  │  Step 3: Hable filmic tone map
  │    _hable_filmic() — 线性 HDR [0, ∞) → [0, 1]
  │    ProPhoto 宽色域下 desaturation 更渐进（见 primer §3）
  │
  │  Step 4: 色域转换 + 传递函数
  │    a. ProPhoto → sRGB 色域映射（3×3 矩阵 + gamut clipping）
  │    b. linear → sRGB 传递函数
  │
  │
  │  ★ 从这里开始：色域 = sRGB，编码 = sRGB gamma ★
  │
  ▼  ImageData(SRGB, [0, 1])
  │
  │  ┌─────────────────────────────────────────────┐
  │  │  sRGB 空间                                    │
  │  │  所有 AI 模型的训练数据都是 sRGB               │
  │  └─────────────────────────────────────────────┘
  │
  ▼  [order=20] denoise 🔲 未实现
  │  processors/denoise.py → DenoiseProcessor
  │  AI 降噪（NAFNet）
  │
  ▼  [order=30] horizon 🔲 未实现
  │  processors/horizon.py → HorizonProcessor
  │  水平线校正
  │
  ▼  [order=40] white_balance 🔲 未实现
  │  processors/white_balance.py → WhiteBalanceProcessor
  │  白平衡残差修正（AI WB 或手动色温偏移）
  │
  ▼  [order=50] color_grade 🔲 未实现
  │  processors/color_grade.py → ColorGradeProcessor
  │  色彩分级（3DLUT AI）
  │
  ▼  [order=60] auto_crop 🔲 未实现
  │  processors/auto_crop.py → AutoCropProcessor
  │  智能构图裁切（YOLOv8）
  │
  ▼  io.save_image() → output.jpg
```

## JPEG / HEIF 路径

```
photo.jpg / .heic
  │
  ▼  io._load_standard() 或 io._load_heif()
  │  Pillow / pillow-heif 加载，归一化到 float32 [0, 1]
  │
  ▼  ImageData(SRGB, [0, 1])
  │
  ▼  [order=5]  lens_correct — 跳过（supported_inputs=["raw"]）
  ▼  [order=10] raw_develop — 跳过（supported_inputs=["raw"]）
  │
  ▼  [order=11] exposure_adjust ✅
  │  processors/exposure_adjust.py → ExposureAdjustProcessor.process()
  │  _mild_exposure_adjust() — 轻度 S 曲线，微提暗部微压高光
  │
  ▼  [order=20+] 与 RAW 路径汇合，进入相同的 sRGB 后处理流水线
  │  denoise → horizon → white_balance → color_grade → auto_crop
  │
  ▼  io.save_image() → output.jpg
```

## 汇合点

RAW 和 JPEG/HEIF 通过各自的前端处理到达统一的 sRGB [0, 1] 空间：

- **RAW** → `raw_develop`（order=10）：auto_ev + Hable filmic + linear_to_srgb
- **JPEG/HEIF** → `exposure_adjust`（order=11）：轻度 S 曲线

从 order=20 开始，所有格式在同一个 sRGB 空间里经过相同的后处理器。

## 色彩空间分区

### ProPhoto 线性域（rawpy → sRGB 转换之前）

rawpy 输出到 sRGB 转换之前的全部操作都在线性 ProPhoto RGB 空间。宽色域保护饱和色不被裁切，线性域保证物理运算精确。

| 阶段 | 色彩空间依赖 |
|------|-------------|
| lens_correct — 暗角/TCA/畸变 | 乘法/几何变换，色彩空间无关 |
| auto_ev — 曝光补偿 | 线性域乘法，色彩空间无关 |
| hable_filmic — tone map | 受色域影响：ProPhoto 下 desaturation 更渐进 |

### sRGB（色域转换 + 传递函数之后）

tone map 后经过色域映射（ProPhoto → sRGB）和传递函数（linear → sRGB gamma）两步进入 sRGB。后续 AI 模型训练数据均为 sRGB，全程 sRGB 工作。

## 处理器状态总表

| order | 处理器 | 文件 | 核心函数 | 格式 | 状态 |
|-------|--------|------|----------|------|------|
| 5 | lens_correct | `processors/lens_correct.py` | `_apply_corrections()` | RAW | ✅ 已实现 |
| 10 | raw_develop | `processors/raw_develop.py` | `_compute_auto_ev()`, `_hable_filmic()` | RAW | 🔧 需确认 |
| 11 | exposure_adjust | `processors/exposure_adjust.py` | `_mild_exposure_adjust()` | JPG/HEIF/PNG/TIFF | ✅ 已实现 |
| 20 | denoise | `processors/denoise.py` | — | any | 🔲 未实现 |
| 30 | horizon | `processors/horizon.py` | — | any | 🔲 未实现 |
| 40 | white_balance | `processors/white_balance.py` | — | any | 🔲 未实现 |
| 50 | color_grade | `processors/color_grade.py` | — | any | 🔲 未实现 |
| 60 | auto_crop | `processors/auto_crop.py` | — | any | 🔲 未实现 |
| 70 | super_res | `processors/super_res.py` | — | any | 🔲 未实现（不在默认步骤） |
| 80 | content_remove | `processors/content_remove.py` | — | any | 🔲 未实现（不在默认步骤） |

状态说明：✅ 已实现，不需修改 ｜ 🔧 已实现，需未来确认/调整 ｜ 🔲 未实现

## 导出

`io.save_image()` 控制最终输出：

| 维度 | 选项 | 控制位置 |
|------|------|----------|
| **色域** | sRGB（Web）、AdobeRGB（印刷）、ProPhoto（归档） | `save_image()` 参数 / `shotline.toml [output]` |
| **编码** | sRGB gamma、gamma 1.8、linear | 由色域隐含决定 |
| **格式** | JPEG（8-bit）、TIFF（8/16-bit）、PNG（8/16-bit） | `shotline.toml [output] format` |
| **位深** | 8-bit（JPEG）、16-bit（TIFF/PNG 归档） | 由格式决定 |

当前实现：统一输出 sRGB 8-bit JPEG。其他组合待后续实现。
