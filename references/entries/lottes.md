# Timothy Lottes / AMD Cauldron Tonemapper

- Shader: `shaders/reference/lottes.wgsl`.
- Algorithm author: Timothy Lottes, AMD.
- Author's official GDC 2016 presentation: https://gpuopen.com/download/GdcVdrLottes.pdf
- Presentation snapshot SHA-256: `8dfe2a69b21cc06bad448e12e90f990a2e2f3e4cd56185ccfd8072614c4505ca`.
- Exact port source: AMD Cauldron's `ColToneB`, `ColToneC`, `ColTone`, `AMDTonemapper` (full max-RGB and nonlinear crosstalk implementation).
- Upstream repository: https://github.com/GPUOpen-LibrariesAndSDKs/Cauldron
- Pinned revision: `b92d559bd083f44df9f8f42a6ad149c1584ae94c`.
- Exact implementation: https://github.com/GPUOpen-LibrariesAndSDKs/Cauldron/blob/b92d559bd083f44df9f8f42a6ad149c1584ae94c/src/VK/shaders/tonemappers.glsl
- Source file SHA-256: `59e456c2c1b06e550cec5fe44aab45fff1ef5dd81db807297791f83fc404517d`.
- License: MIT. Original source-file AMD 2018 copyright notice and repository license retained in `THIRD_PARTY_LICENSES/references/lottes/`.
- Local upstream files (ignored): `third_party/reference_sources/cauldron/` and `third_party/reference_sources/lottes/GdcVdrLottes.pdf`.
- Retrieved: 2026-10-08.

## Transform contract and fidelity

This is the official Cauldron variant, not a simplified per-channel Lottes curve. It preserves tone mapping of max(R,G,B), RGB ratios, and nonlinear channel crosstalk. The original SDR constants are retained: `hdrMax=16`, `contrast=2`, `shoulder=1`, `midIn=midOut=0.18`, `crosstalk=4`, `saturation=2`, `crossSaturation=32`, white=1. The coefficient B expression is algebraically rearranged to the two-anchor solution given on presentation page 47; C and the transform are unchanged.

Input is scene-linear Rec.709 after AP0/D60 to Rec.709/D65 conversion and negative-channel clipping by the workbench adapter. Output is display-linear Rec.709; presentation encoding is external. The max-RGB epsilon remains 1e-6 as upstream. This pinned preset is SDR and ignores `headroom`; AMD's comments suggest changing several anchors for HDR but do not specify a complete fixed HDR preset, so no invented HDR parameterization is introduced. Beyond `hdrMax`, the original math can slightly exceed output 1; the reference preserves this and the shared SDR presenter clips for display.

Independent CPU fixtures evaluate the original nested upstream `ColToneB` expression rather than the rearranged WGSL expression. They verify both the 18% anchor and the 16-to-1 peak anchor, and cover crosstalk in warm, blue and pure-red highlights. See `references/tests/simple_reference_vectors.json` and its `.py` generator.

GPU fixtures also record `input_ap0` after rgba16float quantization and the reconstructed Rec.709 RGB. The primary expected output evaluates the independent author mathematics on that reconstructed input; unquantized expected values remain as metadata. This separates shader-port error from the input texture precision.
