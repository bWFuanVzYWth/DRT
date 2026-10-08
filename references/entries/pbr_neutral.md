# Khronos PBR Neutral

- Shader: `shaders/reference/pbr_neutral.wgsl`.
- Algorithm/source owner: The Khronos Group; developed by Emmett Lalish.
- Upstream repository: https://github.com/KhronosGroup/ToneMapping
- Pinned revision: `180b1a7bddec33f73fe41712a2963cc3ad8e5547`.
- Exact implementation: https://github.com/KhronosGroup/ToneMapping/blob/180b1a7bddec33f73fe41712a2963cc3ad8e5547/PBR_Neutral/pbrNeutral.glsl
- Source SHA-256: `9378f88ad7533e5fb5e10c7e75e9409d2a941089801f2b0b74df6ff17dc7d742`.
- Algorithm specification: https://github.com/KhronosGroup/ToneMapping/blob/180b1a7bddec33f73fe41712a2963cc3ad8e5547/PBR_Neutral/README.md
- License: shader code Apache-2.0; upstream written documentation is CC-BY-4.0. Shader license retained in `THIRD_PARTY_LICENSES/references/pbr_neutral/LICENSE.txt`.
- Local upstream checkout (ignored): `third_party/reference_sources/pbr_neutral/`.
- Retrieved: 2026-10-08.

## Transform contract and fidelity

The official transform takes nonnegative scene-linear Rec.709 RGB and produces display-linear Rec.709 in [0,1]. Original constants are retained: Fresnel offset 0.04, compression start 0.76, desaturation 0.15. This is an SDR reference; `headroom` is unused.

The workbench adapter converts AP0/D60 to Rec.709/D65 and clips negative Rec.709 components before invoking the reference math. That is an explicit adapter for the wider AP0 input, not part of the original tone mapper. The upstream algorithm assumes its input is already in Rec.709 and performs no gamut mapping. Encoding and output presentation belong to the shared workbench wrapper. No look adjustment or exposure compensation is added.

Independent double-precision fixtures, including both 0.08 toe and 0.76 compression boundaries and saturated highlights, are stored in `references/tests/simple_reference_vectors.json`; regenerate/check with `references/tests/simple_reference_vectors.py`. Fixture inputs are Rec.709 before the AP0 adapter and outputs are display-linear.

GPU fixtures also record `input_ap0` after rgba16float quantization and the reconstructed Rec.709 RGB. The primary expected output evaluates the independent author mathematics on that reconstructed input; unquantized expected values remain as metadata. This separates shader-port error from the input texture precision.
