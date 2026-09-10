# DRT Bench

用于开发、调试和比较显示渲染变换（Display Rendering Transform，DRT）的交互式工作台。使用 Rust、wgpu 和 WGSL，在统一的图像输入与显示环境中观察色调映射、色域映射和颜色表现。

## 功能

- 切换不同 DRT，调整曝光与算法参数，实时查看映射结果。
- 结合中性灰轴曲线、颜色分布和数值异常可视化检查输出；灰轴曲线独立于当前图片及其曝光。
- 提供内置测试图、图片加载和文件夹浏览，支持 EXR、HDR、PNG、JPEG 和 WebP。
- 自动热重载 DRT 着色器；编译失败时保留上一条有效管线。

## 输入与显示

工作台以 scene-linear ACES2065-1（AP0）作为统一输入。EXR/HDR 按这一约定读取；普通 SDR 图片按 sRGB 解码并转换到 AP0。

SDR 输出使用 sRGB。支持 HDR 的 DRT 可通过线性 scRGB 显示，亮度以系统 SDR 参考白为基准；HDR 是否可用取决于系统、显示器和窗口表面能力，不可用时回落到 SDR。

## DRT 命名与结构

项目中的自定义 DRT 按“颜色处理模型 + 曲线类型”命名。`RGB` 指通过中性轴 inset/outset 进入和离开的虚拟 RGB 工作坐标；三款 RGB DRT 均逐通道应用曲线，并提供 HSV 色相回拉。HSV 是曲线之后的调整步骤。

| 名称 | 主要结构 | 着色器 |
| --- | --- | --- |
| None | 无色调压缩，执行显示转换与范围裁切 | [none_drt.wgsl](shaders/none_drt.wgsl) |
| Oklab Reinhard | 在 Oklab L³ 上应用线性段与 Reinhard 肩部，再沿固定 Oklab 色相方向柔性压缩色度 | [oklab_reinhard.wgsl](shaders/oklab_reinhard.wgsl) |
| AgX-S2O3 | 保留原始 AgX-S2O3 结构的参考移植 | [agx_s2o3.wgsl](shaders/agx_s2o3.wgsl) |
| RGB Log Sigmoid | 整个有效输入范围进入 log2 坐标，以独立 toe/shoulder 构成逐通道 S 曲线，输出显示信号 | [rgb_log_sigmoid.wgsl](shaders/rgb_log_sigmoid.wgsl) |
| RGB Reinhard | 逐通道线性段与 Reinhard 肩部，输出线性光后做 sRGB 编码 | [rgb_reinhard.wgsl](shaders/rgb_reinhard.wgsl) |
| RGB Log Shoulder | 逐通道线性段与 log1p sigmoid 肩部连续衔接，输出线性光后做 sRGB 编码 | [rgb_log_shoulder.wgsl](shaders/rgb_log_shoulder.wgsl) |

`AgX-S2O3` 是 linlin 对原始参考的移植，结构与原始参考相同，但并非 AgX 原作者的原始 Python 实现。本仓库的 WGSL 版本移植自 linlin 的 GLSL 参考移植，具体来源版本记录在文件头；linlin 的署名指移植实现，原始算法来源仍为 AgX-S2O3。该参考保留原名。

`RGB Log Sigmoid` 沿用 AgX 的对数 S 曲线结构，并使用本项目的默认参数、HDR 扩展和 HSV 色相回拉；`RGB Log Shoulder` 使用 AgX 形式的 sigmoid 肩部，并在线性段之后引入 log1p 坐标。结构来源保留在说明中，名称用于区分当前实现。

## 运行

需要支持 Rust edition 2024 的工具链和可供 wgpu 使用的图形驱动，无需额外安装着色器编译器或 Vulkan SDK。

```powershell
cargo run
cargo run -- image.exr
cargo run -- --folder path/to/images
```

无输入时显示内置测试图。可附加 `--analysis` 打开分析视图，或 `--show-anomalies` 启用数值异常显示。

`F3` 打开图片，`F4` 打开文件夹，`F5` 重新编译当前 DRT，`Esc` 退出。颜色分布视图支持左键拖动旋转、右键复位。

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
