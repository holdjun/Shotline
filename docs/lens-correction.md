# 镜头光学校正（Lens Correction）

## 概述

`lens_correct` processor（order=5）在线性空间对 RAW 图像做镜头光学校正，依赖 [lensfunpy](https://github.com/letmaik/lensfunpy)（lensfun 的 Python binding）。

仅适用于 `source_format="raw"` 的 ImageData。JPG/HEIF 已在相机内完成镜头校正，无需重复处理。

## 校正流程与顺序

```
ImageData(LINEAR, raw_loader.exif)
    │
    ▼
1. lensfun DB 查询: camera_make/model → lens profile
    │
    ▼
2. apply_color_modification() — 暗角校正（像素亮度修正）
    │
    ▼
3. apply_subpixel_geometry_distortion() → cv2.remap() — TCA + 畸变校正
    │
    ▼
4. auto_crop — 裁切几何校正产生的黑边
    │
    ▼
ImageData(LINEAR, 光学校正后)
```

**暗角必须先于几何校正**：暗角修改像素亮度值，几何校正做插值重映射。若先做几何校正再暗角，边缘区域的插值像素会得到错误的暗角补偿。

**TCA 与畸变合并**：当 `correct_tca=true` 时，始终通过 `apply_subpixel_geometry_distortion()` 同时校正 TCA 和畸变（per-channel remap），无论 `correct_distortion` 的值——因为 TCA 的 per-channel 重映射天然包含几何畸变校正，分离没有意义。只有在 `correct_tca=false` 且 `correct_distortion=true` 时，才走 `apply_geometry_distortion()` 单通道 remap。

## lensfun 数据库

### 覆盖范围

- **Bundled DB**（lensfunpy 1.18.0 捆绑 lensfun 0.3.4）：948 相机 / 1304 镜头
- **更新 DB**（lensfun 官方）：1045+ 相机 / 1582+ 镜头，持续增长

### 自动更新机制

仓库内 `data/lensfun-db/` 存放 lensfun 官方最新 DB XML 文件：

1. GitHub Action 每周检查 `https://lensfun.github.io/db/version_1.tar.bz2`
2. 通过 SHA256 hash 判断是否有更新
3. 有更新 → 自动创建 PR，包含新增 camera/lens 摘要
4. `lens_correct` 自动加载仓库内 DB（`lensfunpy.Database(paths=[...])`）

### 版本兼容性

仓库内 DB 使用 `version_1` 格式（`<lensdatabase version="1">`），与当前 lensfunpy 完全兼容。`Database(paths=[...])` 在 bundled DB 基础上叠加仓库内 DB 中的条目（新增 camera/lens 合并进来，同名条目以 paths 中的版本为准）。

## 依赖与降级策略

| 依赖 | 用途 | 缺失时行为 |
|------|------|------------|
| `lensfunpy>=1.16` | 镜头校正核心 | processor status=UNAVAILABLE，自动跳过 |
| `opencv-python-headless>=4.8` | `cv2.remap()` 几何重映射 | 仅做暗角校正，跳过几何校正 |
| `exifread>=3.0`（间接，raw_loader） | 上游 EXIF 解析 | EXIF 提取失败，lens_correct 跳过 |

| 运行时情况 | 行为 |
|------------|------|
| EXIF 提取失败 | 跳过，metadata 记录原因 |
| 镜头不在 lensfun 数据库 | 跳过，metadata 记录 camera/lens |
| focal_length 或 aperture 缺失 | 跳过，metadata 记录原因 |

## 配置参数

```toml
[processor_params.lens_correct]
correct_distortion = true      # 几何畸变校正
correct_vignetting = true      # 暗角校正
correct_tca = true             # 横向色差校正（启用时自动包含畸变校正）
distance = 10.0                # 拍摄距离（米），默认 10m
```

`auto_crop`（几何校正后裁切黑边）始终自动执行，不可单独关闭。

## 元数据输出

`metadata["lens_correct"]`：

```python
{
    "camera": "ILCE-7M3",
    "lens": "Sony FE 24-70mm f/2.8 GM",
    "crop_factor": 1.0,
    "focal_length": 35.0,
    "aperture": 2.8,
    "distance": 10.0,
    "corrections": {
        "vignetting": True,       # 仅在实际执行时出现
        "tca": True,              # 仅在实际执行时出现
        "distortion": True,       # 仅在实际执行时出现
        "auto_crop": {"top": 2, "bottom": 2, "left": 3, "right": 3}
    }
}
```

`corrections` 中各 key 仅在对应校正实际执行时才出现，禁用的校正不会出现（而非设为 `False`）。

跳过时：

```python
{
    "skipped": "camera/lens not in lensfun database",
    "camera": "ILCE-7M3",
    "lens": "Unknown Lens"
}
```
