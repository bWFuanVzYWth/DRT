# ACES 2.0 Curve

在完整的 ACES 2.0 WGSL 移植上，只替换标量亮度曲线的研究版本。选择器名称为 `ACES 2.0 Curve`，着色器为 [aces_20_curve.wgsl](aces_20_curve.wgsl)，支持独立控件、分屏、SDR/HDR 和热重载。

基础实现：[官方 ACES 2.0 移植记录](../../references/entries/aces_20.md)。上游固定为 [aces-core 069b0bc3e1f6c62820f19fdae2fecec3f4fc0f80](https://github.com/aces-aswf/aces-core/tree/069b0bc3e1f6c62820f19fdae2fecec3f4fc0f80)，保留 Apache-2.0 和 Contributors to the ACES Project 署名。这款实验不属于 Academy 的官方输出预设。

## 唯一的算法改动

`a20_tonemap_compress` 仍从输入 J 求回场景亮度 `linear = a20_j_y(J) / 100`。将原来的 modified Michaelis-Menten 加 flare 标量公式替换为：

```text
T(x) = k*x                                         x <= j
T(x) = peak - A*(1 + q/p)^(-p)                    x > j
A = peak - k*j
q = k*(x-j)/A
```

输出峰值依旧读取原 ACES 表的 `reference_data[102] / 100`。曲线在接点的值与一阶导数连续，高光渐近显示峰值；接点附近用等价 Taylor 展开避免浮点抵消。

默认 `Linear slope = 1`、`Compression start = 0.18`、SDR `Highlight reach = 10 EV`。reach 表示标量曲线达到当前显示峰值 98% 的场景亮度 EV，以 18% 灰为基准。三个控件独立保存，复用现有长肩部参数求解。

这会改变灰阶位置：默认线性段的 18% 灰输出约为 `0.18`，官方 SDR ACES 2 曲线约为 `0.10013`。该差异来自新的亮度曲线，不是额外曝光补偿。

## 保留的处理

- AP0/AP1 转换、AP1 输入限制、RGB ↔ JMh 模型与白点处理。
- 原来的色度归一化、随明度变化的色度扩展/压缩和 hue-dependent reach。
- 原来的 cusp-based 色域映射、输出颜色模型与 Rec.709/D65 显示转换。
- 原来的 host 初始化和全部 1913 个参考浮点值，包括 max J、mid/focus、cusp/reach/gamma 表。
- 原来的显示编码、异常可视化与峰值限制。

色度和色域算法继续接受新曲线给出的明度，因此彩色像素会随曲线改变。原 ACES 2.0 参考项保留原公式。

原 ACES AP1 输入限制也保留，SDR 上限为 1024。reach 描述这个限制之前的数学曲线；若其目标输入超过该限制，完整输出会先遇到原输入上限。默认 10 EV 的目标输入在限制以内。

## 验证

独立检查将新曲线函数移除、恢复原标量公式后，ACES 颜色处理正文与参考移植逐字相同。GPU 验证使用恢复原公式的实验 shader 对照原参考，以确认差异只来自标量曲线；另检查灰阶、SDR/HDR、三个独立控件、分屏状态和热重载。

RTX 4090 / Vulkan 上全套 73 项测试通过，包括原 ACES 官方参考向量；Clippy、格式检查与 Release 构建通过。

```powershell
cargo +stable test --locked aces_curve -- --include-ignored --test-threads=1 --nocapture
```
