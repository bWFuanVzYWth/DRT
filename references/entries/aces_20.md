# ACES 2: complete Output Transform a2.v1

- Official core: [aces-aswf/aces-core](https://github.com/aces-aswf/aces-core), commit `069b0bc3e1f6c62820f19fdae2fecec3f4fc0f80` (ACES 2.0 release content, pinned on 2026-10-08).
- Official presets: [aces-aswf/aces-output](https://github.com/aces-aswf/aces-output), commit `6d8f9071a67b044bac0fbcb3d51ad0543f065e66`.
- Local checkouts: `third_party/reference_sources/aces_20/` and `third_party/reference_sources/aces_output/` (ignored).
- Ports: `shaders/reference/aces_20.wgsl` and `src/aces2_data.rs`.
- License: Apache-2.0; copyright Contributors to the ACES Project. [Pinned license](https://github.com/aces-aswf/aces-core/blob/069b0bc3e1f6c62820f19fdae2fecec3f4fc0f80/LICENSE), retained in `THIRD_PARTY_LICENSES/references/aces_20/LICENSE`.

Permanent source references:

- [OutputTransform.ctl](https://github.com/aces-aswf/aces-core/blob/069b0bc3e1f6c62820f19fdae2fecec3f4fc0f80/lib/Lib.Academy.OutputTransform.ctl).
- [Tonescale.ctl](https://github.com/aces-aswf/aces-core/blob/069b0bc3e1f6c62820f19fdae2fecec3f4fc0f80/lib/Lib.Academy.Tonescale.ctl).
- [Utilities.ctl](https://github.com/aces-aswf/aces-core/blob/069b0bc3e1f6c62820f19fdae2fecec3f4fc0f80/lib/Lib.Academy.Utilities.ctl).
- [Rec.709 / D65 100nit sRGB preset](https://github.com/aces-aswf/aces-output/blob/6d8f9071a67b044bac0fbcb3d51ad0543f065e66/d65/srgb/Output.Academy.Rec709-D65_100nit_in_Rec709-D65_sRGB-Piecewise.ctl).

Input is scene-linear ACES2065-1 / AP0. Output is display-linear Rec.709 / D65 in units where 1 = 100nit. The default headroom 1 selects the official 100nit Rec.709-limited sRGB preset; common output encoding supplies the sRGB EOTF inverse. `scale_white` is false and limiting / encoding primaries both use Rec.709 / D65.

HDR headroom sets the parametric transform's peak to `100 * headroom` nits, up to the upstream model's 10000nit range. This is the same complete ACES 2 algorithm with a Rec.709 limiting gamut and linear scRGB representation. It is a workbench parameterization, **not a claim that Rec.709 1000nit is a separately published Academy preset**. Tone scale, J maximum, chroma compression, cusp and reach tables, and upper-hull fitting are regenerated for each peak.

The shader analytically implements AP1 input clamping, the Hellwig / CAM16-based RGB↔JMh model, modified Michaelis-Menten tonescale with flare, hue-dependent chroma expansion / compression, cusp-based gamut compression, and the output model inversion. This is not a tone-curve fit or a 3D LUT.

CPU initialization ports the official `init_JMhParams`, `init_TSParams`, reach gamut binary searches, reach / limiting corner search, sorted corner hue sampling, cusp search, smoothing, and upper-hull gamma fitting. It uses f64 during initialization and packs f32 for WGSL; numerical precision consequently differs from a CTL implementation that evaluates intermediate data in float. The gamut shader searches the whole cusp table, equivalent to the upstream narrowed search window. The upstream gamma `interpolation_weight` returning `h - h_lo` is deliberately preserved, including its lack of interval-width normalization.

Storage buffer layout is 1913 f32 values: input JMh 0..41, limiting JMh 41..82, tonescale 82..93, compression constants 93..103, padded reach M 103..465, padded J/M/h cusp table 465..1551, and padded upper-hull inverse gamma 1551..1913. Matrices use column-major storage. Full details are documented beside `generate(headroom)` in `src/aces2_data.rs`.

Validation includes model inversion and neutrality, finite host parameters for 100 / 200 / 1000 / 4000 / 10000nit, and the project's shader / GPU checks. Independent vectors are tracked in `references/validation/aces_reference_vectors.json`; `generate_aces_ocio_vectors.py` uses the official ASWF **PyOpenColorIO 2.6.0** CPU `FIXED_FUNCTION_ACES_OUTPUT_TRANSFORM_20` with the same Rec.709 / D65 primaries and peak luminance. The OCIO fixed function implements the JMh core; the fixture generator explicitly adds the official CTL AP1 input clamp before evaluating it, deriving conversion matrices independently from the published AP0 / AP1 chromaticities. There are 33 samples at each of five peaks (100 / 200 / 1000 / 4000 / 6400nit), quantized to float16 before oracle evaluation to match the source texture. This is the numerical oracle used by the reference GPU suite. The OCIO port and its initialization precision are independent of this Rust / WGSL implementation, so agreement is checked with floating-point and output-half rounding tolerance.

The actual GPU runtime passed all 198 ACES 1.3 / ACES 2 vectors on the NVIDIA RTX 4090 / Vulkan adapter with `cargo +stable test --locked reference_aces_ports -- --ignored --nocapture`. This also compiled all registered reference / research pipelines. The encoded-output tolerance is `0.0015 + 0.001 * abs(expected)`, accounting for f32 evaluation and f16 output storage.

A CTL-to-C++ diagnostic evaluator was attempted but its runtime was not validated; it is **not** used as evidence of correctness and is not included among tracked tools. The accepted numerical oracle is the official OpenColorIO implementation above.
