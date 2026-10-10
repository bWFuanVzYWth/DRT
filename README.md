# DRT Bench

用于开发、调试和比较显示渲染变换（Display Rendering Transform，DRT）的交互式工作台。使用 Rust、wgpu 和 WGSL，在统一的图像输入与显示环境中观察色调映射、色域映射和颜色表现。

## 功能

- 切换不同 DRT，调整曝光与算法参数，实时查看映射结果。
- 提供 12 个第三方参考 DRT、7 个独立研究实现与 None 基线；每份参考移植登记固定远程来源、许可、默认预设和适配差异。
- 启用 Compare 对比模式，在同一张图片上用可拖动分割线比较左右 DRT。
- 结合中性灰轴曲线、颜色分布和数值异常可视化检查输出；曲线横轴为对数输入 EV，纵轴为线性显示亮度，独立于当前图片及其曝光。
- 提供内置测试图、图片加载和文件夹浏览，支持 EXR、HDR、PNG、JPEG 和 WebP。
- 自动热重载 DRT 着色器；编译失败时保留上一条有效管线。

## 输入与显示

工作台以 scene-linear ACES2065-1（AP0）作为统一输入。EXR/HDR 按这一约定读取；普通 SDR 图片按 sRGB 解码并转换到 AP0。

SDR 输出使用 sRGB。支持 HDR 的 DRT 可通过线性 scRGB 显示，亮度以系统 SDR 参考白为基准；HDR 是否可用取决于系统、显示器和窗口表面能力，不可用时回落到 SDR。

中性灰轴曲线横轴保留相对于 18% 灰的 −16 至 +18 EV；纵轴从精确黑 0 到当前 DRT 的显示峰值，单位为 SDR 参考白倍数。SDR 的范围为 0–1，HDR 按当前输出峰值缩放，刻度为峰值的 0%、25%、50%、75%、100% 并显示实际亮度值。未压缩的线性参考在这一坐标系下为指数曲线，到达显示峰值后停止绘制。

sRGB / Oklab 颜色分布图按当前 DRT 的输出范围自动缩放，并适配画布比例，保留高于 SDR 白的 HDR 点。sRGB 图的外框表示完整输出范围，HDR 下另显示内部的 SDR 参考立方体；图上标注范围相对于 SDR 白的倍数。仅支持 SDR 的 DRT 保持 0–1× 范围。

## DRT 命名与结构

项目中的自定义 DRT 按“颜色处理模型 + 曲线类型”命名。`RGB` 指通过中性轴 inset/outset 进入和离开的虚拟 RGB 工作坐标；两款 RGB DRT 均逐通道应用曲线，并提供 HSV 色相回拉。HSV 是曲线之后的调整步骤。

第三方参考放在 [shaders/reference/](shaders/reference/)，研究实现放在 [shaders/research/](shaders/research/)。参考包括 AgX-S2O3、ACES 1.3、ACES 2、Blender AgX、Filmic、AMD LPM、Hable、Khronos PBR Neutral、Lottes、OpenDRT、Uchimura 和 Narkowicz ACES Filmic Fit；完整预设、不可变源码链接和验证记录见 [参考登记](references/README.md)。上游原始资料下载到项目内 `third_party/reference_sources/`，不进入 Git；运行所需移植 LUT 进入 Git。

| 名称 | 主要结构 | 着色器 |
| --- | --- | --- |
| None | 无色调压缩，执行显示转换与范围裁切 | [none_drt.wgsl](shaders/research/none_drt.wgsl) |
| AgX-S2O3 | 保留原始 AgX-S2O3 结构的参考移植 | [agx_s2o3.wgsl](shaders/reference/agx_s2o3.wgsl) |
| ACES 2.0 Curve | 在完整 ACES 2.0 中只替换标量亮度曲线，保留 JMh、色度压缩及色域映射 | [aces_20_curve.wgsl](shaders/research/aces_20_curve.wgsl) |
| Oklab Log Sigmoid | 将 RGB Log Sigmoid 的解析线性暗部与 log2 sigmoid 肩部应用到 Oklab L³，解码回线性亮度后执行 Oklab 色度压缩 | [oklab_log_sigmoid.wgsl](shaders/research/oklab_log_sigmoid.wgsl) |
| Oklab Reinhard | 在 Oklab L³ 上应用线性段与 Reinhard 肩部，再沿固定 Oklab 色相方向柔性压缩色度 | [oklab_reinhard.wgsl](shaders/research/oklab_reinhard.wgsl) |
| Oklab ACES-inspired | Oklab L³ 的线性低段与渐近长肩部，结合固定色相方向的柔性色度边界处理 | [oklab_aces.wgsl](shaders/research/oklab_aces.wgsl) |
| Oklab Neutral | 在 BT.709 最大通道上应用线性段与长肩部，以 Oklab root-LMS 四次式路径趋向白 | [oklab_neutral.wgsl](shaders/research/oklab_neutral.wgsl) |
| RGB Log Sigmoid | 在 log2 坐标中构造解析线性光低段，中灰处与 sigmoid 肩部相切，逐通道输出显示信号 | [rgb_log_sigmoid.wgsl](shaders/research/rgb_log_sigmoid.wgsl) |
| RGB Reinhard | 逐通道线性段与 Reinhard 肩部，输出线性光后做 sRGB 编码 | [rgb_reinhard.wgsl](shaders/research/rgb_reinhard.wgsl) |

`AgX-S2O3` 是 linlin 对原始参考的移植，结构与原始参考相同，但并非 AgX 原作者的原始 Python 实现。本仓库的 WGSL 版本移植自 linlin 的 GLSL 参考移植，具体来源版本记录在文件头；linlin 的署名指移植实现，原始算法来源仍为 AgX-S2O3。该参考保留原名。

`RGB Log Sigmoid` 保留 AgX 形式的对数 sigmoid 肩部，暗部改为在 log2 坐标中解析构造的线性光编码曲线，并保留 HDR 扩展和 HSV 色相回拉。色域压缩可调，默认为 0.04；色相回拉使用单一 `Hue retention` 参数，默认 0.5，在全亮度范围保持相同强度。

`RGB Log Sigmoid` 在分段点 `j` 以下使用 `F(u) = sRGB_OETF(k × j × 2^((u − pivot) × dynamicRange))`，显示解码后恰为 `k × x`。由同一表达式推导接点输出值和肩部切线，使两段的值与一阶导数连续；零输入直接得到零，极暗输入没有有限的 log 黑位截断。默认 `Linear slope` 为 1、`Compression start` 为 0.18，保留原默认 sigmoid 肩部。分段点移动时，`Highlight reach` 仍以 18% 灰为基准，最小 reach 随斜率和分段点约束以保持有效肩部。

这里的线性保证针对标量曲线与中性灰轴；inset/outset 和 HSV 处理仍可能使彩色暗部偏离 `None`，亮像素中的低值通道也会受新曲线影响。原 `Linear shadows` 开关与过渡混合已移除，`Shadow reach`、`Toe power`、`Mid contrast` 和独立中灰输出控制仅保留于 `AgX-S2O3` 参考。

`Oklab Log Sigmoid` 与 `Oklab Reinhard` 使用 SDR 色域边界，分别保存各自的曲线参数。

`Oklab Log Sigmoid` 与 `RGB Log Sigmoid` 共用标量曲线参数和默认值：Linear slope = 1、Compression start = 0.18、Highlight reach = 6.5 EV、Shoulder power = 5.2；各自独立保存设置。移植路径为 `L³ → log2 曲线 → sRGB 解码 → 立方根 → Oklab L`，相同曲线参数下中性灰轴与 RGB 版的 SDR 曲线对应。颜色部分使用现有 Oklab 固定色相方向的柔性色度压缩，因此彩色高光表现与逐通道 RGB 曲线不同；不使用 RGB inset/outset 或 HSV 色相回拉。

`Oklab ACES-inspired` 是借鉴 ACES 2 分离亮度与色彩强度处理的原创实验，不是官方 ACES 移植。统一输入转换后，在 Oklab 中依次处理 `L³ → 长肩部亮度映射 → 固定色相的柔性色度边界 → 显示 RGB`，支持 SDR/HDR。亮度曲线的低段仍为 `Linear slope × L³`，灰轴保持线性；彩色边缘从暗部起连续保护，避免在肩部接点突然压缩色度。色域边缘颜色不保证与 None 完全一致。

其肩部为 `T = k·j + A·[1 − (1 + q/p)^(−p)]`，其中 `q = k·(L³ − j)/A`、`A = peak − k·j`；在分段点与线性段保持值和一阶导数连续，接点附近的二阶导数有限。`p = 1` 对应渐近 Reinhard 肩部，没有有限的输入白点或 log 黑位截断。界面直接以 `Highlight reach` 控制到达当前输出峰值 98% 所需的输入 EV，默认 SDR 为 +10 EV（相对于 18% 灰）。值越小越快收向白，值越大高光延伸越长；该数值描述肩部长度，并非裁切点。内部在 `p = 0.25–8` 的范围反推肩部幂次，控件的可调范围随线性斜率、分段点与当前 SDR/HDR 峰值实际计算；分段点已达到目标亮度时需降低 `Compression start` 才能调节。

颜色阶段复用旧 Oklab 的固定色相边界和 soft-min，使用固定默认系数，不继承另一款 DRT 的色度控件。以 `L'/peak^(1/3)` 归一化查询边界，再恢复实际 HDR 明度和色度，使映射后的正常颜色已在目标显示范围内，避免逐 RGB 通道裁切造成品红/蓝色偏移。

颜色阶段在全部明度范围使用同一条固定色相 soft-min 映射：`outputSaturation = softMin(inputSaturation, displayCap, roundingPower)`。近白区 0.90–0.97 的源饱和度分级混合已经撤回，避免在这一段切换映射而使趋白轨迹转弯；近白的不同饱和度可能再次趋近同一显示边界。长肩部、10 EV 默认和显示边界的蓝色保护保持原样，没有额外 `1 − w²` 或 `1 − L^12` 衰减。原高光去色和纯色保护控件继续删除。

`Oklab Neutral` 是独立保存参数的另一款原创实验。输入从 AP0 转为线性 BT.709 后适配为非负 RGB，以最大通道作为亮度标量；默认 `Linear slope = 1`、`Compression start = 0.6`、SDR `Highlight reach = 10 EV`，高光延伸同样表示达到当前峰值 98% 的输入 EV。肩部复用上述标量曲线，低段保持线性 RGB 增益；颜色在 LMS 立方根坐标中趋向中性白，近似保持 Oklab 色相。以肩部进度 `P = 1 − (1 + q/p)^(−p)` 计算固定四次式插白权重 `H(P) = P³·(3 − 2P)`，不增加独立饱和度滑条。蓝色附近的非凸边界可能产生微小负 RGB，采用共同 RGB 抬升后统一归一化；输入的负值限制属于非负 BT.709 工作域适配，不是宽色域压缩。支持 SDR/HDR，参数与其他 DRT 独立。

`ACES 2.0 Curve` 直接基于完整 ACES 2.0 移植，只把场景 Y 的标量亮度曲线换成线性暗部与长肩部。默认斜率 1、压缩起点 0.18、SDR reach 10 EV；JMh、白点、色度处理、色域映射及原始参考表均保留，独立保存三个曲线参数。灰阶位置也随曲线改变，默认 18% 灰输出约 0.18。完整来源与验证见 [实验记录](shaders/research/aces_20_curve.md)。

高光褪色的三次、四次和五次贝塞尔替换实验保存在 [贝塞尔实验记录](shaders/research/oklab_neutral_bezier.md)；可独立运行 GPU 审计。控制点能扩大白端开口并减轻部分混色反弹，但真实样片的暖色边缘和明度回落仍有代价，当前应用继续使用上述四次式。

## 参数约定

选择器按第三方参考与研究实现分组，None 保留为基线。七款研究 DRT 各自独立保存设置，相同含义的曲线参数使用相同命名。AgX-S2O3 保留参考参数与控制，包括默认 0.2 的 inset。新增参考以各自登记的原版默认预设运行，不统一改写中灰或高光颜色行为。

| 参数 | 适用范围 | 默认值 / 含义 |
| --- | --- | --- |
| Linear slope | 七款研究 DRT | 1；直接表示线性暗部的输出增益，替代 Reinhard 原来的间接 Input scale |
| Compression start | 七款研究 DRT | Oklab Neutral 为 0.6，其余为 0.18；scene-linear RGB / Oklab L³ / BT.709 最大通道的线性段与肩部分段点 |
| Highlight reach | 七款研究 DRT | RGB Reinhard、Oklab ACES-inspired、Oklab Neutral 与 ACES 2.0 Curve 默认 SDR 为 10 EV，其余为 6.5 EV；以 18% 灰为基准，最小值随曲线约束。三款实验以当前显示峰值的 98% 为目标，SDR/HDR 可调范围随实际峰值计算 |
| Shoulder power | 两款 Log Sigmoid | 5.2；三款实验肩部幂次由 Highlight reach 反推，不提供独立幂次控件；原有 Reinhard 不包含独立的幂次控制 |
| Gamut compression | 两款 RGB | 0.04；虚拟 RGB inset 系数，可调范围 0–0.8 |
| Hue retention | 两款 RGB | 0.5；全亮度范围一致的 HSV 色相修复强度 |
| Highlight chroma power | 原有两款 Oklab | 12；`1 − L^p` 的高光色度衰减幂次 |
| Gamut rounding power | 原有两款 Oklab | 4；黑端、白端色域边界相接时的 soft-min 幂次 |
| Endpoint chroma power | 原有两款 Oklab | 32；靠近黑白两端的色度压缩幂次 |
| Midtone chroma power | 原有两款 Oklab | 16；Oklab L = 0.5 处的色度压缩幂次，向两端平滑过渡 |

分段点上限为 `0.99 / Linear slope`，保证接点输出低于 SDR 白。Reinhard 支持从 0 开始压缩；Log Sigmoid 的分段点必须为正，其下限随斜率计算，使相切肩部能够在 20 EV 的高光范围内到达白点。调整斜率或分段点时会同步约束高光范围。

原有两款 Oklab 的四个色度控制来自既有固定系数，默认值保留原效果，且分别保存设置。它们不改变灰轴曲线；色相沿 Oklab 原方向保持，因此没有额外的 HSV 修复步骤。色彩空间矩阵和色域边界求解的数值修正继续使用既有常量。

## 运行

需要支持 Rust edition 2024 的工具链和可供 wgpu 使用的图形驱动，无需额外安装着色器编译器或 Vulkan SDK。

```powershell
cargo run
cargo run -- image.exr
cargo run -- --folder path/to/images
```

无输入时显示内置测试图。可附加 `--analysis` 打开分析视图，或 `--show-anomalies` 启用数值异常显示。

内置测试图保留原有默认 7 条带，并增加 `Color trajectories (61 bands)` 趋白轨迹图，可通过 `File` 菜单或侧栏 `Test patterns` 选择。新图取线性 sRGB 的 R/G/B 各为 0、0.25、0.5、0.75、1，且至少一个通道为 1 的全部 61 种组合；每条带横向连续增加场景亮度倍率，从精确黑到 `2^14 = 16384×`，覆盖暗部、SDR 与 HDR 高光。颜色在乘亮度倍率后转换为统一的 scene-linear AP0 输入，便于比较各 DRT 的不同颜色趋白轨迹。

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

`cargo +stable test --locked -- --include-ignored` 包含参考的 420 组独立数值对照，以及全部 19 个 DRT 的 361 种左右对比组合。实验另验证低段线性、SDR/HDR 长肩部、颜色趋白轨迹，以及各参数对对比画面的独立更新。参考数据生成和来源复现方法见 [验证说明](references/README.md#来源与复现)。

外部测试图可放入已忽略的 `test-assets/`，不随项目分发。

## 许可

本项目采用 [GNU General Public License v3.0](LICENSE)（`GPL-3.0-only`）。

第三方代码及 LUT 的原始许可/署名记录保留在 [THIRD_PARTY_LICENSES/references/](THIRD_PARTY_LICENSES/references/)；来源和未明确的资产许可状态在各参考条目中独立登记。
