# DRT Bench

以 Oklab DRT 为主、AgX-S2O3 与 AgX-HSV 为参照，并包含独立色调映射实验的开发测试环境。Oklab DRT 原先借助 `C:\WorkSpace\drt-bench` 协同设计和验证；AgX-S2O3 从独立 MIT 项目 `C:\WorkSpace\AgX\agx.glsl` 移植，不包含 `drt-bench` 中的其它 DRT。GPU 路径是：

```text
Slang 2025.13+ → SPIR-V 1.3 → wgpu 30 → Vulkan / Direct3D 12 / Metal
                                      ↘ egui 0.36 UI / winit presentation
```

输入约定为 scene-linear ACES2065-1（AP0）。所有 DRT 先输出到 display-encoded extended-sRGB `RGBA16F` 中间纹理；窗口支持时最终 pass 将其解码为线性 scRGB，并通过 wgpu 30 的 `Rgba16Float + ExtendedSrgbLinear` surface 做实际 HDR 输出，`1.0` 始终表示系统 SDR reference white。系统或显示器没有暴露 HDR surface 时自动回落到 SDR sRGB。Oklab 路径包含肩部曲线、色域 cusp、Halley 修正和 saturation soft-min；AgX-S2O3 路径按原始 OCIO 链路先将 AP0 转为线性 BT.709、钳制负分量，再执行可调的解析 sigmoid。AgX-S2O3 的曲线结果已经是显示编码，不再叠加 OETF。不会引入 OpenDRT、Skibidi 或色彩立方体。

## 环境要求

- 支持 edition 2024 的 Rust 工具链；
- `slangc` 2025.13 或更高版本；
- Windows、Linux 或 macOS 上可供 wgpu 使用的图形驱动。

Vulkan SDK 1.3.296+ 自带 Slang。构建脚本依次查找 `SLANGC`、`PATH` 和 `VULKAN_SDK/Bin`，并拒绝低于 2025.13 的版本。

## 运行

```powershell
cargo run
cargo run -- frame.exr
cargo run -- --analysis
cargo run -- --folder C:\images
cargo run -- frame.exr --show-anomalies
```

也可在 UI 中打开 EXR、HDR、PNG、JPEG 或 WebP。浮点 EXR/HDR 被视为 scene-linear AP0；普通 SDR 图片按 sRGB 解码并从 Rec.709 转为 AP0。

- `F3`：打开图片；
- `F4`：打开包含测试图片的文件夹；
- `F5`：重新执行 Slang → SPIR-V 编译；
- `Esc`：退出；
- 编辑 `shaders/none_drt.slang`、`shaders/oklab_drt.slang`、`shaders/agx_s2o3.slang`、`shaders/agx_hsv.slang` 或 `shaders/reinhard_gamut.slang` 后会分别自动热重载；编译失败时保留对应 DRT 的上一条有效管线。

UI 用常驻按钮在 `None`、`Oklab`、`AgX-S2O3`、`AgX-HSV` 与 `Reinhard-Gamut` 间一键切换。所有路径共用 `-20..+20 EV` 曝光；`None` 仅执行 AP0 到 Rec.709 和 IEC sRGB 编码，不做 tone mapping，可作为显示参考；`0.5..2.0` 高光渐近值（默认 `1.0`）仅用于 Oklab。Oklab 在亮度映射后、色域 cusp 计算前主动重排高光 hue：黄、青、洋红是吸引轴，红、绿、蓝是保持不动的分界轴，压缩量按映射后的 Oklab lightness 从可调 onset 平滑增长到白端强度；默认 onset 为 40%，白端压缩为 80%。AgX-S2O3 与 AgX-HSV 拥有互不覆盖的 tone-scale 配置。S2O3 作为原始 SDR 参考，默认参数为 `-10/+6.5 EV`、输出中灰 `0.5`、枢轴对比度 `2.0`、toe/shoulder power `3.0/3.25` 和 gamut compression `0.2`。

`Reinhard-Gamut` 是独立 SDR 实验：AP0 转到线性 Rec.709 后，先把虚拟原色向外移动，使当前坐标按可调比例收缩到中性轴；在该坐标中逐通道应用 `scale*x/(1+scale*x)`，随后使用解析逆变换还原到 Rec.709 并执行标准 sRGB 编码。默认虚拟色域扩张为 20%，默认 input scale 为 `1/(1-0.18) = 1.2195122`，因此 18% 中性灰严格映射回线性 0.18。色域变换本身可逆，Reinhard 夹在正逆变换之间才会改变彩色高光轨迹；此实验暂不随 HDR headroom 扩展。

AgX-HSV 是同一套 SDR/HDR DRT，不另设 HDR 变体。界面的 HDR headroom 使用显示器实时报告的 `peak / SDR white`，也可以手动降低；设为 `1.0×` 时输出峰值、曲线端点和全部 SDR tone-scale 系数数值退化为当前 SDR 行为。大于 `1.0×` 时中灰、toe、枢轴斜率和 SDR reference white 均不动，只按 extended-sRGB 输出峰值同比延长 shoulder 的输入 EV 区间与输出区间，让近线性高光继续延伸并更慢趋白。AgX-HSV 的输出中灰采用精确的 `sRGB OETF(0.18) = 0.46135613`，默认枢轴斜率 `2.46064` 与 16.5-stop 分配下的 `None` 局部斜率一致，toe power `1.55` 拟合可见暗部的中性灰阶；shoulder power 暂用 `5.2`。已知逐通道强肩部会在高饱和到低饱和的高光过渡中产生感知断层，当前暂时接受，后续需要把色彩轨迹与 tone curve 解耦。gamut compression 保持 `0.05`。AgX-HSV 会在 2.2 编码 RGB 域比较原始 Rec.709 与 AgX 输出的 HSV 色相，并沿最短色相角修复，同时保持 AgX 输出的 value。黑端与白端色相保持度默认分别为 100% 和 50%，每个像素按相对于目标峰值归一化的 AgX HSV value 在两端之间线性插值；修复后的 saturation 直接钳制到 `0..1`。界面会根据 EV 容量和输出中灰动态限制最低合法斜率，避免曲线进入复数或 NaN 域。没有输入图片时使用内置的 AP0 HDR 色条和 16-stop 曝光扫描图。

左侧固定显示当前 DRT 的中性灰轴函数图。横轴是 scene-linear AP0 中性灰相对 18% 灰的输入 EV，纵轴是映射后 display-linear 中性灰相对 18% 灰的输出 EV，两轴均为 `log2` 域；同一坐标内的蓝灰色对角线表示不做 tone mapping 的线性参考。曲线使用独立生成的 512 点 AP0 灰轴并直接经过当前 DRT 的同一条 GPU shader，不读取当前图片，也不受图片曝光控制影响；切换 DRT、调整曲线参数或 HDR headroom 时实时更新。左侧控制区整体可滚动，函数图不会挤掉其余参数。

顶部可在完整映射图和“映射图 + 色彩分布”两个界面间切换；分布视图支持 display-encoded sRGB 与 Oklab 坐标空间，并可拖动旋转。可视化直接用 `vertex_index` 将映射结果的每个像素变成一个点：不随机抽样、不降采样，标题会显示实际完整点数。点云使用带安全边界的正交投影，旋转不会让点越过透视近裁剪面。

sRGB 分布带有单位立方体参照，并以红、绿、蓝高亮从黑点出发的 RGB 基向量；Oklab 暂不绘制参照。左键拖动旋转点云，右键单击恢复默认视角。

“Show anomalies”默认关闭：有限输出钳制到显示范围，`+Inf` 清洗为白、`-Inf` 清洗为黑，NaN 或混合 `±Inf` 显示为洋红。启用后，NaN、`+Inf`、`-Inf`、混合 `±Inf`、有限负值和有限超范围值分别使用洋红、黄、青、橙、蓝、红显示；界面会同时显示图例。

打开文件夹后，左侧显示该目录中的 EXR、HDR、PNG、JPEG 和 WebP 文件。缩略图在后台依次生成；单击条目会在后台加载主图并立即切换 DRT 与色彩分布，列表本身不会因大图解码而失去响应。

## 验证

```powershell
cargo fmt -- --check
cargo test
cargo clippy --all-targets -- -D warnings
```

`build.rs` 每次着色器变更都会生成并嵌入 SPIR-V；运行时 wgpu 还会验证模块、绑定和计算管线。可用 `WGPU_BACKEND=vulkan`、`dx12`、`metal` 或 `gl` 强制选择后端。

## 测试图资产

仓库只包含自行生成的 AP0 HDR 测试图，不随代码再分发 ACES 外部图集。需要 ACES reference/candidate frames 时应由使用者从上游单独取得，放入已忽略的 `test-assets/`；只有确认具体素材的再分发授权后，才考虑将其纳入发布物。

## 许可

尚未选择项目整体的开源协议。在协议确定前，本仓库不构成整体开源许可或再分发授权；`Cargo.toml` 设置了 `publish = false`，避免误发布到 crates.io。AgX-S2O3 移植部分沿用上游 MIT License，完整版权与许可文本见 `THIRD_PARTY_LICENSES/AgX-S2O3.txt`。
