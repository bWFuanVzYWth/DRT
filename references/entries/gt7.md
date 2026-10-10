# GT7 Tone Mapping

- Shader: `shaders/reference/gt7.wgsl`.
- Authors: Kenichiro Yasutomi, Kentaro Suzuki and Hajime Uchimura, Polyphony Digital Inc.
- Official publication index: https://www.polyphony.co.jp/publications/
- Publication-index snapshot SHA-256: `552036d73158a72f9cbce5fc2f008a85b5f0b07819ddc1072e927efc3e59d5c8`.
- Official SIGGRAPH 2025 presentation, *Driving Toward Reality: Physically Based Tone Mapping and Perceptual Fidelity in Gran Turismo 7*: https://s3.amazonaws.com/gran-turismo.com/pdi_publications/s2025_PBS_Physically_Based_Tone_Mapping_GT7.pdf
- PDF snapshot SHA-256: `8f6e31f1850aa1a178a19ca9b0df4317f248ccb9e3f4bf5a861b4adb9a1fcc06` (32,467,812 bytes; 243 pages). The GT7 operator and sample are described on pages 122-157; the complete code is supplied separately below.
- Official course page linked by Polyphony: https://blog.selfshadow.com/publications/s2025-shading-course/
- Course-page snapshot SHA-256: `31ffa1958ab2bcb3676d888a9ca30b87bd973f64c2829f18a73ff4feca6be503`.
- Author-provided C++ implementation, linked as **[code]** on the official course page: https://blog.selfshadow.com/publications/s2025-shading-course/pdi/supplemental/gt7_tone_mapping.cpp
- Source version: 1.0, initial release 2025-08-10.
- Source snapshot SHA-256: `94df7e7e310b9423ebb5aaaa29416f376ebbae66e0948fb259f8688314b29822` (19,193 bytes).
- Retrieved: 2026-10-11.
- Local upstream files (ignored): `third_party/reference_sources/gt7/` contains the exact C++ sample, publication/course pages, original PDF, extracted slide text and cache manifest. None of these upstream downloads is tracked in Git.
- License: MIT, copyright (c) 2025 Polyphony Digital Inc.; retained in the shader header and `THIRD_PARTY_LICENSES/references/gt7/LICENSE.txt`.

## Operator fidelity

This port implements the complete default ICtCp branch of the published sample. It is distinct from the older Uchimura / Gran Turismo Sport tone curve already in this workbench.

The GT Tone Mapping V2 component curve retains alpha `0.25`, gray point `0.538`, linear section `0.444` and toe strength `1.280`. The source Rec.2020 input is converted to ICtCp using the sample's ST-2084 functions and matrices. The ICtCp intensity of the component-mapped color supplies the new intensity; the original input chroma is multiplied by the original fade curve, with intensity-ratio endpoints `0.98` and `1.16`. Reconstruction preserves the sample's PQ input clamp and nonnegative Rec.2020 output. The final RGB result is `0.4 * component-mapped + 0.6 * reconstructed`, clipped to the original target peak. The optional Jzazbz branch, scene grading and eye adaptation are not part of this reference selection.

Initialization constants are computed on the CPU when the target headroom changes, following the original `initializeParameters` and `initializeCurve` methods. Eight scalar values are uploaded through the existing reference storage buffer: peak, target ICtCp intensity, shoulder A/B/C, output unit conversion and two padding values. These are curve constants, with no sampled LUT or interpolation.

## Workbench adapter and luminance units

The source declares scene/frame-buffer `1.0 = 100 cd/m²` and SDR paper white `250 cd/m²`. Its exact SDR configuration initializes a target frame-buffer peak of `2.5` and then multiplies the completed output by `0.4`. The workbench retains this SDR result, including its native exposure and middle-gray placement.

For effective workbench headroom `H = clamp(selected headroom, 1, 40)`, initialize the original HDR operator at `250 * H cd/m²` (source frame-buffer peak `2.5 * H`) and convert its returned frame-buffer RGB to units of the original SDR paper white by dividing by `2.5`. At `H = 1` this is exactly the sample's SDR configuration. This uses the author's genuine display-peak parameter, and provides continuous SDR/HDR behavior; it does not stretch an already tone-mapped SDR image. The workbench still treats output `1.0` as its relative SDR-white unit, so the source's physical paper-white convention does not reconfigure the actual monitor's SDR-white setting.

The sample documents HDR target luminances from 250 to 10,000 cd/m²; that corresponds to workbench headroom from 1 to 40 with this adapter. When the workbench's overall HDR headroom exceeds 40, the GT7 reference retains its supported 10,000 cd/m² target and effective output peak 40. This preserves the source's documented domain and ST-2084 reconstruction limit.

Input is converted from workbench AP0/D60 to Rec.2020/D65 using the shared chromatic adaptation and standard Rec.709/Rec.2020 conversion. Negative Rec.2020 components are clipped before entering the published sample's nonnegative PQ domain. Output is converted from the sample's linear Rec.2020 to display-linear Rec.709. The raw reference transform preserves resulting Rec.709 excursions; the shared display adapter performs the final display clipping and sRGB encoding. An additional game-specific Rec.2020-to-Rec.709 gamut mapper is not invented.

## Validation

The port is compared against the downloaded official C++ implementation, compiled independently of the WGSL. Fixtures include the author's example colors, the power-toe and shoulder transitions, chroma-fade transitions, near-white colors, saturated highlights and SDR/HDR display peaks. Input AP0 quantization is recorded separately so the GPU comparison uses the values actually stored in the rgba16float source texture. See `references/validation/generate_gt7_vectors.py`, `references/validation/gt7_reference_vectors.json` and `src/reference_validation.rs`.

All 222 compiled-official vectors passed on NVIDIA RTX 4090 / Vulkan, for selected headroom 1, 2, 4, 16, 40 and 64 (the last uses effective peak 40). The GT7 comparison decodes the actual f16 sRGB output and compares display-linear values with a budget of `0.0005 * effective_peak + 0.002 * clipped_expected_linear`. The peak term accounts for FP32 PQ/SFU error and Rec.2020-to-Rec.709 cancellation; the relative term accounts for f16 encoded-output quantization. Other reference ports retain their existing encoded-output tolerance.

An independent evaluation of the same formulas in double precision confirmed that the published C++ FP32 implementation itself loses precision at high peaks. At headroom 40, the bright-cyan fixture's red channel is 0.1687666 in official FP32 C++, 0.1809354 in double precision, and 0.186665 after decoding the GPU's f16 output. The WGSL retains the official formulas rather than tuning them to one platform's floating-point results. The largest tested error uses 88% of the declared budget.
