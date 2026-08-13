# Oklab DRT Bench

单 Oklab DRT 开发测试环境。Oklab DRT 原先借助 `C:\WorkSpace\drt-bench` 协同设计和验证；本工程只保留需要长期维护的 Oklab 实现，不迁移其它 DRT。GPU 路径是：

```text
Slang 2025.13+ → SPIR-V 1.3 → wgpu 30 → Vulkan / Direct3D 12 / Metal
                                      ↘ egui / eframe 0.36 UI
```

输入约定为 scene-linear ACES2065-1（AP0），输出为 SDR display-encoded sRGB。工程包含 Oklab 映射、肩部曲线、色域 cusp、Halley 修正和 saturation soft-min；不会引入 AgX、OpenDRT、Skibidi 或色彩立方体。

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
```

也可在 UI 中打开 EXR、HDR、PNG、JPEG 或 WebP。浮点 EXR/HDR 被视为 scene-linear AP0；普通 SDR 图片按 sRGB 解码并从 Rec.709 转为 AP0。

- `F3`：打开图片；
- `F5`：重新执行 Slang → SPIR-V 编译；
- `Esc`：退出；
- 编辑 `shaders/oklab_drt.slang` 后会自动热重载；编译失败时保留上一条有效管线。

UI 暴露参考工程 Oklab 路径的两个运行时参数：`-20..+20 EV` 曝光，以及 `0.5..2.0` 高光渐近值（默认 `1.1`）。没有输入图片时使用内置的 AP0 HDR 色条和 16-stop 曝光扫描图。

顶部可在完整映射图和“映射图 + 色彩分布”两个界面间切换；分布视图支持 display-encoded sRGB 与 Oklab 坐标空间，并可拖动旋转。可视化直接用 `vertex_index` 将映射结果的每个像素变成一个点：不随机抽样、不降采样，标题会显示实际完整点数。点云使用带安全边界的正交投影，旋转不会让点越过透视近裁剪面。

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

尚未选择开源协议。在协议确定前，本仓库不构成开源许可或再分发授权；`Cargo.toml` 设置了 `publish = false`，避免误发布到 crates.io。
