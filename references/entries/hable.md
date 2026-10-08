# Hable / Uncharted 2 Filmic

- Shader: `shaders/reference/hable.wgsl`.
- Algorithm author: John Hable.
- Author's original explanation and HLSL: https://filmicworlds.com/blog/filmic-tonemapping-operators/ (2010-05-05).
- Author page snapshot SHA-256: `c7ca3ae7c60b76b997e5c1575aaddbd55ae0f4fcc271a4d9e7ea1ca25fb36c33`.
- Port source: AMD's official MIT implementation in Cauldron.
- Upstream repository: https://github.com/GPUOpen-LibrariesAndSDKs/Cauldron
- Pinned revision: `b92d559bd083f44df9f8f42a6ad149c1584ae94c`.
- Exact implementation: https://github.com/GPUOpen-LibrariesAndSDKs/Cauldron/blob/b92d559bd083f44df9f8f42a6ad149c1584ae94c/src/VK/shaders/tonemappers.glsl (`Uncharted2TonemapOp`, `Uncharted2Tonemap`).
- Source file SHA-256: `59e456c2c1b06e550cec5fe44aab45fff1ef5dd81db807297791f83fc404517d`.
- License: MIT. The source file's AMD 2018 notice and repository license are retained in `THIRD_PARTY_LICENSES/references/hable/`.
- Local upstream files (ignored): `third_party/reference_sources/cauldron/` and `third_party/reference_sources/hable/filmic-tonemapping-operators.html`.
- Retrieved: 2026-10-08.

## Transform contract and fidelity

The reference evaluates nonnegative linear Rec.709 RGB using A=0.15, B=0.50, C=0.10, D=0.20, E=0.02, F=0.30, white point W=11.2. AMD's implementation and Hable's demo apply exposure bias 2.0 before the curve; this multiplier is retained in addition to the workbench's user exposure. Consequently scene input 5.6 maps to output white 1. The demo's image-specific multiplication by 16 is omitted, because it is the user's exposure rather than part of the operator.

The adapter converts AP0/D60 to Rec.709/D65 and clips negative Rec.709 channels. Output is display-linear Rec.709. The author demo applies a 2.2 power display encoding afterwards; this port exposes the linear curve result and the workbench applies its common sRGB/scRGB presentation. This is an SDR reference and `headroom` is unused. Values above the selected white point can exceed 1, exactly as in the official implementation; final SDR display clipping is a presentation step, not a change to the reference curve.

Independent double-precision fixtures include 18% gray, the exact 5.6 scene-input white point, saturated RGB values and over-range highlights in `references/tests/simple_reference_vectors.json`. The fixture generator is `references/tests/simple_reference_vectors.py`.

GPU fixtures also record `input_ap0` after rgba16float quantization and the reconstructed Rec.709 RGB. The primary expected output evaluates the independent author mathematics on that reconstructed input; unquantized expected values remain as metadata. This separates shader-port error from the input texture precision.
