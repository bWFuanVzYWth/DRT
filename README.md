# DRT Bench

用于开发、调试和比较显示渲染变换（Display Rendering Transform，DRT）的交互式工作台。使用 Rust、wgpu 和 WGSL，在统一的图像输入与显示环境中观察色调映射、色域映射和颜色表现。

## 功能

- 切换不同 DRT，调整曝光与算法参数，实时查看映射结果。
- 启用 Compare 对比模式，在同一张图片上用可拖动分割线比较左右 DRT。
- 结合中性灰轴曲线、颜色分布和数值异常可视化检查输出；灰轴曲线独立于当前图片及其曝光。
- 提供内置测试图、图片加载和文件夹浏览，支持 EXR、HDR、PNG、JPEG 和 WebP。
- 自动热重载 DRT 着色器；编译失败时保留上一条有效管线。

## 输入与显示

工作台以 scene-linear ACES2065-1（AP0）作为统一输入。EXR/HDR 按这一约定读取；普通 SDR 图片按 sRGB 解码并转换到 AP0。

SDR 输出使用 sRGB。支持 HDR 的 DRT 可通过线性 scRGB 显示，亮度以系统 SDR 参考白为基准；HDR 是否可用取决于系统、显示器和窗口表面能力，不可用时回落到 SDR。

## DRT 命名与结构

项目中的自定义 DRT 按“颜色处理模型 + 曲线类型”命名。`RGB` 指通过中性轴 inset/outset 进入和离开的虚拟 RGB 工作坐标；两款 RGB DRT 均逐通道应用曲线，并提供 HSV 色相回拉。HSV 是曲线之后的调整步骤。

| 名称 | 主要结构 | 着色器 |
| --- | --- | --- |
| None | 无色调压缩，执行显示转换与范围裁切 | [none_drt.wgsl](shaders/none_drt.wgsl) |
| AgX-S2O3 | 保留原始 AgX-S2O3 结构的参考移植 | [agx_s2o3.wgsl](shaders/agx_s2o3.wgsl) |
| Oklab Log Sigmoid | 将 RGB Log Sigmoid 的解析线性暗部与 log2 sigmoid 肩部应用到 Oklab L³，解码回线性亮度后执行 Oklab 色度压缩 | [oklab_log_sigmoid.wgsl](shaders/oklab_log_sigmoid.wgsl) |
| Oklab Reinhard | 在 Oklab L³ 上应用线性段与 Reinhard 肩部，再沿固定 Oklab 色相方向柔性压缩色度 | [oklab_reinhard.wgsl](shaders/oklab_reinhard.wgsl) |
| RGB Log Sigmoid | 在 log2 坐标中构造解析线性光低段，中灰处与 sigmoid 肩部相切，逐通道输出显示信号 | [rgb_log_sigmoid.wgsl](shaders/rgb_log_sigmoid.wgsl) |
| RGB Reinhard | 逐通道线性段与 Reinhard 肩部，输出线性光后做 sRGB 编码 | [rgb_reinhard.wgsl](shaders/rgb_reinhard.wgsl) |

`AgX-S2O3` 是 linlin 对原始参考的移植，结构与原始参考相同，但并非 AgX 原作者的原始 Python 实现。本仓库的 WGSL 版本移植自 linlin 的 GLSL 参考移植，具体来源版本记录在文件头；linlin 的署名指移植实现，原始算法来源仍为 AgX-S2O3。该参考保留原名。

`RGB Log Sigmoid` 保留 AgX 形式的对数 sigmoid 肩部，暗部改为在 log2 坐标中解析构造的线性光编码曲线，并保留 HDR 扩展和 HSV 色相回拉。色域压缩可调，默认为 0.04；色相回拉使用单一 `Hue retention` 参数，默认 0.5，在全亮度范围保持相同强度。

`RGB Log Sigmoid` 在分段点 `j` 以下使用 `F(u) = sRGB_OETF(k × j × 2^((u − pivot) × dynamicRange))`，显示解码后恰为 `k × x`。由同一表达式推导接点输出值和肩部切线，使两段的值与一阶导数连续；零输入直接得到零，极暗输入没有有限的 log 黑位截断。默认 `Linear slope` 为 1、`Compression start` 为 0.18，保留原默认 sigmoid 肩部。分段点移动时，`Highlight reach` 仍以 18% 灰为基准，最小 reach 随斜率和分段点约束以保持有效肩部。

这里的线性保证针对标量曲线与中性灰轴；inset/outset 和 HSV 处理仍可能使彩色暗部偏离 `None`，亮像素中的低值通道也会受新曲线影响。原 `Linear shadows` 开关与过渡混合已移除，`Shadow reach`、`Toe power`、`Mid contrast` 和独立中灰输出控制仅保留于 `AgX-S2O3` 参考。

两款 Oklab DRT 使用 SDR 色域边界，分别保存各自的曲线参数。

`Oklab Log Sigmoid` 与 `RGB Log Sigmoid` 共用标量曲线参数和默认值：Linear slope = 1、Compression start = 0.18、Highlight reach = 6.5 EV、Shoulder power = 5.2；各自独立保存设置。移植路径为 `L³ → log2 曲线 → sRGB 解码 → 立方根 → Oklab L`，相同曲线参数下中性灰轴与 RGB 版的 SDR 曲线对应。颜色部分使用现有 Oklab 固定色相方向的柔性色度压缩，因此彩色高光表现与逐通道 RGB 曲线不同；不使用 RGB inset/outset 或 HSV 色相回拉。

## 参数约定

选择顺序为 None、AgX-S2O3、Oklab Log Sigmoid、Oklab Reinhard、RGB Log Sigmoid、RGB Reinhard。四款自定义 DRT 的曲线参数使用相同命名，各自独立保存设置。AgX-S2O3 保留参考参数与控制，包括默认 0.2 的 inset。

| 参数 | 适用范围 | 默认值 / 含义 |
| --- | --- | --- |
| Linear slope | 四款自定义 DRT | 1；直接表示线性暗部的输出增益，替代 Reinhard 原来的间接 Input scale |
| Compression start | 四款自定义 DRT | 0.18；scene-linear RGB / Oklab L³ 的线性段与肩部分段点 |
| Highlight reach | 四款自定义 DRT | RGB Reinhard 为 10 EV，其余为 6.5 EV；以 18% 灰为基准，最小值随曲线约束 |
| Shoulder power | 两款 Log Sigmoid | 5.2；Reinhard 的有理肩部不包含独立的幂次控制 |
| Gamut compression | 两款 RGB | 0.04；虚拟 RGB inset 系数，可调范围 0–0.8 |
| Hue retention | 两款 RGB | 0.5；全亮度范围一致的 HSV 色相修复强度 |
| Highlight chroma power | 两款 Oklab | 12；`1 − L^p` 的高光色度衰减幂次 |
| Gamut rounding power | 两款 Oklab | 4；黑端、白端色域边界相接时的 soft-min 幂次 |
| Endpoint chroma power | 两款 Oklab | 32；靠近黑白两端的色度压缩幂次 |
| Midtone chroma power | 两款 Oklab | 16；Oklab L = 0.5 处的色度压缩幂次，向两端平滑过渡 |

分段点上限为 `0.99 / Linear slope`，保证接点输出低于 SDR 白。Reinhard 支持从 0 开始压缩；Log Sigmoid 的分段点必须为正，其下限随斜率计算，使相切肩部能够在 20 EV 的高光范围内到达白点。调整斜率或分段点时会同步约束高光范围。

Oklab 的四个色度控制来自原有的固定系数，默认值保留原效果，且两款 Oklab 分别保存设置。它们不改变灰轴曲线；色相沿 Oklab 原方向保持，因此没有额外的 HSV 修复步骤。色彩空间矩阵和色域边界求解的数值修正继续使用既有常量。

## 运行

需要支持 Rust edition 2024 的工具链和可供 wgpu 使用的图形驱动，无需额外安装着色器编译器或 Vulkan SDK。

```powershell
cargo run
cargo run -- image.exr
cargo run -- --folder path/to/images
```

无输入时显示内置测试图。可附加 `--analysis` 打开分析视图，或 `--show-anomalies` 启用数值异常显示。

`F3` 打开图片，`F4` 打开文件夹，`F5` 重新编译当前 DRT，`Esc` 退出。颜色分布视图支持左键拖动旋转、右键复位。

勾选顶部的 `Compare` 后，原有 DRT 选择控制左侧画面，`Right DRT` 选择右侧画面。两侧共用输入图片、曝光、显示范围与异常显示，并使用各 DRT 已保存的参数。拖动图片中的竖直分割线查看同一位置的映射差异；双击分割线或点击 `Center divider` 可恢复居中，`Swap sides` 可交换两侧并调整另一款 DRT 的参数。对比模式同样适用于分析视图，其中灰轴曲线和颜色分布对应左侧 DRT；`F5` 会重新编译两侧使用的 DRT。

## 开发与验证

DRT 的 WGSL 实现位于 [shaders/](shaders/)，工作台与 GPU 调度位于 [src/](src/)。着色器随程序嵌入，由 wgpu 创建管线；开发时编辑源文件可自动热重载。各算法的参数定义和数学细节以实现为准。

```powershell
cargo fmt -- --check
cargo test
cargo clippy --all-targets -- -D warnings
```

有可用 GPU 时，运行 `cargo test gpu::validation -- --ignored` 检查着色器执行、SDR/HDR 输出、异常值和热重载恢复。

外部测试图可放入已忽略的 `test-assets/`，不随项目分发。

## 许可

本项目采用 [GNU General Public License v3.0](LICENSE)（`GPL-3.0-only`）。
