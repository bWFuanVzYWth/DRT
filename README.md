# DRT Bench

以 Oklab DRT 为主、AgX-S2O3 为参照的开发测试环境。Oklab DRT 原先借助 `C:\WorkSpace\drt-bench` 协同设计和验证；AgX-S2O3 从独立 MIT 项目 `C:\WorkSpace\AgX\agx.glsl` 移植，不包含 `drt-bench` 中的其它 DRT。GPU 路径是：

```text
Slang 2025.13+ → SPIR-V 1.3 → wgpu 30 → Vulkan / Direct3D 12 / Metal
                                      ↘ egui / eframe 0.36 UI
```

输入约定为 scene-linear ACES2065-1（AP0），输出为 SDR display-encoded sRGB。Oklab 路径包含肩部曲线、色域 cusp、Halley 修正和 saturation soft-min；AgX-S2O3 路径按原始 OCIO 链路先将 AP0 转为线性 BT.709、钳制负分量，再执行 16.5-stop 解析 sigmoid。AgX-S2O3 的曲线结果已经是显示编码，直接写入 sRGB 呈现链路，不再叠加 OETF。不会引入 OpenDRT、Skibidi 或色彩立方体。

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
- 编辑 `shaders/oklab_drt.slang` 或 `shaders/agx_s2o3.slang` 后会分别自动热重载；编译失败时保留对应 DRT 的上一条有效管线。

UI 可在 `Oklab` 与 `AgX-S2O3` 间即时切换。两者共用 `-20..+20 EV` 曝光；`0.5..2.0` 高光渐近值（默认 `1.1`）仅用于 Oklab。没有输入图片时使用内置的 AP0 HDR 色条和 16-stop 曝光扫描图。

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
