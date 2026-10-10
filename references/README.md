# 第三方参考 DRT 登记

参考实现与研究实现分开存放。原始上游资料位于项目内的 `third_party/reference_sources/`，整个 `third_party/` 已被 Git 忽略；移植代码、运行数据、来源登记、许可证/署名记录和验证工具进入 Git。运行程序不需要上游 checkout、Python 或 OpenColorIO。

## 目录

- `shaders/reference/`：13 个参考 DRT、共享输入/显示适配层和 `assets/` 内的运行 LUT。
- `shaders/research/`：Oklab ACES-inspired 与 RGB Reinhard 两个研究质量基线；None 显示转换位于 `shaders/none_drt.wgsl`。
- `shaders/analysis/`：颜色分布与灰轴曲线可视化。
- `references/entries/`：逐项登记固定远程版本、输入/输出、原始默认值、适配差异和验证证据。
- `THIRD_PARTY_LICENSES/references/`：保留上游许可和资产署名；未知的许可明确标记，不借用其他实现的许可证。
- `references/validation/` 与 `references/tests/`：独立上游数值、可复现生成器和数据打包记录。

## 实现清单

| 登记 | 上游参考 | 输出预设 |
| --- | --- | --- |
| [AgX-S2O3](entries/agx_s2o3.md) | linlin 原始解析 GLSL，固定提交；现有移植迁入参考目录 | SDR，保留现有参数 |
| [ACES 1.3 RRT + ODT](entries/aces_13.md) | Academy v1.3 CTL | sRGB 100nit dim，SDR |
| [ACES 2](entries/aces_20.md) | Academy a2.v1 CTL，完整解析算法与主机色域表 | Rec.709/D65，100 × headroom nits |
| [Blender AgX](entries/blender_agx.md) | Blender 固定 OCIO 配置和原始 57³ LUT | Base Rec.1886，无 look，SDR |
| [Filmic](entries/filmic.md) | Troy Sobotka 原始 65³ LUT、4096 点 Base Contrast 曲线 | SDR |
| [AMD FidelityFX LPM](entries/fidelityfx_lpm.md) | AMD 官方 Setup + Filter + Map | Rec.2020 工作色域 → Rec.709，SDR/scRGB 容器缩放 |
| [Hable / Uncharted 2](entries/hable.md) | 作者公式与 AMD Cauldron 固定源码 | 原始曝光偏置/白点，SDR |
| [Khronos PBR Neutral](entries/pbr_neutral.md) | Khronos 官方 GLSL | SDR |
| [Lottes 2016](entries/lottes.md) | AMD Cauldron 官方 max-RGB + crosstalk | 原始 SDR 参数 |
| [OpenDRT](entries/opendrt.md) | Jed Smith v1.1.0 DCTL | Standard，Rec.709/D65，SDR/HDR |
| [Uchimura / GT](entries/uchimura.md) | Polyphony 官方讲义、作者 Desmos 公式/默认值 | 原公式峰值参数，SDR/HDR |
| [GT7 Tone Mapping](entries/gt7.md) | Polyphony SIGGRAPH 2025 官方曲线与 ICtCp 颜色体积映射 | 原始默认值，SDR/HDR |
| [ACES Filmic Fitted](entries/aces_fitted.md) | Narkowicz 原作者五系数拟合 | SDR；独立于完整 ACES 命名 |

新增移植都接入主画面、左右对比、独立灰轴、颜色分析、异常诊断和 shader 热重载。选择器按 Third-party references / Research transforms 分组，并提供固定远程来源链接。第三方参考使用明确的原版默认预设；研究 DRT 的调参机制保持原样。

## 统一显示约定与保留差异

输入为 scene-linear ACES2065-1/AP0。各参考按其登记的白点/工作色域转换；不强行使用一个不同的矩阵或把所有中灰改成相同亮度。新参考函数返回 display-linear Rec.709；共享适配层处理最终 sRGB/scRGB 编码、范围裁切和异常颜色。

HDR 可配置参考的 `headroom` 范围通常是 1–64。GT7 使用原版 250nit SDR paper white 约定，保留官方 10,000nit 目标上限，对应有效 `headroom` 1–40；更高的全局值按 40 处理。ACES 2/OpenDRT 以 100nit 为数值参考白；线性输出 1 接入系统 SDR 参考白，实际显示绝对亮度由系统设置决定。ACES 2 的 Rec.709 HDR 是官方参数化算法在工作台的配置，不冒称另一份正式发布的 Academy 预设。LPM 的扩展采用已登记的 scRGB 容器缩放。固定 SDR 预设在 HDR 环境中仍使用 SDR 范围。

原版差异也被保留：OpenDRT Standard 的黑位略有抬升，LPM 官方 Soft Gap/crosstalk 会影响高光的通道比例，Hable/Lottes 的原公式可能超出最终显示范围。数值验证针对这些原始行为，而不是靠改公式满足研究 DRT 的统一曲线假设。

## 来源与复现

Git 上游记录使用不可变提交及文件 URL；网页/PDF/Desmos 使用原始 URL、获取日期和本地内容 SHA256。原始文件在忽略目录内保留。LUT 原始 SHA256、矩阵和可重复 float32 打包记录位于 [lut_assets.json](validation/lut_assets.json)。

独立 oracle 共 642 个实际 fp16 输入样本：简易曲线 90、OpenDRT/LPM 66、ACES 198、AgX/Filmic 66、GT7 222。官方 OpenColorIO CPU 实现、作者公式或编译后的上游函数作为参考；GPU 输出与独立结果比较，不仅检查 shader 能否编译。GT7 的高峰值 PQ 浮点消减误差及显示线性比较预算单独登记在其来源条目中。

```powershell
cargo +stable test --locked -- --include-ignored
cargo +stable clippy --locked --all-targets -- -D warnings
cargo +stable fmt --all -- --check
python references/tests/simple_reference_vectors.py --check
python references/validation/generate_lut_assets.py --check
python references/validation/generate_gt7_vectors.py
```

生成器需要明确登记的上游 checkout 和本地验证环境；它们不是运行依赖。OpenColorIO 2.6.0 安装在忽略的 `.cache/reference-python/`。Windows 验证工具使用进程级错误窗口抑制和超时；失效的 CTL-to-C++ 临时验证程序已隔离，不属于跟踪的复现工具。

Blender AgX/Filmic 原始 LUT 在固定来源中未找到独立明确的资产许可文本，登记为 `NOASSERTION` 并保留署名；详情在各自条目和 source notice 中。
