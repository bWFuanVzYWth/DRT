# AMD FidelityFX LPM 1.20200225

- Author: Advanced Micro Devices, Inc.
- Upstream repository: https://github.com/GPUOpen-Effects/FidelityFX-LPM
- Immutable commit: `ed6ecd5b8963d2ec24603809b90ecfa00a1c3614`.
- Reference: [ffx_lpm.h](https://github.com/GPUOpen-Effects/FidelityFX-LPM/blob/ed6ecd5b8963d2ec24603809b90ecfa00a1c3614/ffx-lpm/ffx_lpm.h).
- Original SHA-256: `3e976127b7e0ca119f9f0ecc229498179012fa5cff0026aec279aa6aae98d11e`.
- Look defaults: [the official sample's Rec2020 preset](https://github.com/GPUOpen-Effects/FidelityFX-LPM/blob/ed6ecd5b8963d2ec24603809b90ecfa00a1c3614/sample/src/VK/UI.cpp#L165).
- Local upstream checkout: `third_party/reference_sources/fidelityfx_lpm/` (ignored by Git).
- Ports: `shaders/reference/fidelityfx_lpm.wgsl`, `src/lpm_data.rs`.
- License: MIT, copyright (c) 2017–2019 Advanced Micro Devices, Inc. The full original notice is preserved in `THIRD_PARTY_LICENSES/references/fidelityfx_lpm/LICENSE`.
- Retrieved and ported: 2026-10-08.

This is a direct specialization of **LpmSetup + LpmFilter + LpmMap**, using `LPM_CONFIG_709_2020` and `LPM_COLORS_709_2020`: Rec.2020 D65 working primaries, Rec.709 D65 output, first gamut conversion enabled, soft gamut mapping enabled, final conversion disabled. The CPU port retains the original primary-matrix construction, inverse/multiply, luma normalization, tone constraint solve and first ten float4 control blocks. The shader retains max-RGB ratios, saturation powers, luma-preserving tonescale, soft falloff and the two-stage crosstalk walk back into gamut. No LUT or fitted tone curve replaces the algorithm.

The look values come from the official Rec2020 sample preset: input maximum 16, exposure 4 stops, contrast 0, shoulder contrast 1, saturation (0,0,0), crosstalk (1,1/2,1/32), soft gap 1/32. Shoulder contrast 1 makes the optional shoulder branch an identity, so the port uses its documented fast path. This preset is an explicit reproducible reference choice; LPM is a configurable mapper without one universal final look.

Workbench scene-linear AP0 is adapted through the existing D60-to-D65 Rec.709 conversion, then transformed to Rec.2020 D65. Negative Rec.2020 values are clipped **before** LPM, because the original filter requires nonnegative working RGB for max-RGB ratios and fractional powers. Values above the preset's input maximum are allowed and converge to the output limit. Black returns its continuous black limit before the original reciprocal-of-zero operation; an all-white zero-capacity denominator skips a zero luma addition. These numerical guards leave defined nonzero reference behavior unchanged.

LPM returns linear Rec.709 [0,1]. The final workbench adapter multiplies by display headroom, equivalent to LPM's final `scaleOnly`/scRGB container scaling. Output is **display-linear Rec.709, SDR white = 1, peak = headroom**. The core look is fixed; headroom changes container scale, rather than changing LPM look settings or pretending to run the official HDR10/Rec.2020 PQ output configuration. Final presentation encoding is performed by the workbench.

The storage data starts at float offset 0, with the original 40-float block layout documented in `src/lpm_data.rs`; the unused secondary-conversion fields remain zero. The host setup is regenerated through `generate(headroom)`, although its fixed look constants do not depend on headroom.

Validation compiles the actual upstream `LpmSetup` under `A_CPU`. Its first 40 floats match the Rust setup within **2e-6**. The generator extracts the actual upstream 32-bit `LpmMap` body and translates only CPU types for execution; it does not implement a second handwritten mapper. For the available GPU, 33 AP0 samples at headroom 1/4/10 differed from the compiled upstream by at most **3.63e-6** in linear output. Goldens and regeneration are tracked in `references/validation/hdr_reference_vectors.json` and `generate_hdr_vectors.py`.
