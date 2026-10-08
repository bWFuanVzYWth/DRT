# Narkowicz ACESFilm Fit

- Shader: `shaders/reference/aces_fitted.wgsl`.
- Author: Krzysztof Narkowicz.
- Exact original implementation and permission statement: https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
- Published: 2016-01-06.
- Author page snapshot SHA-256: `d848a767b10280db23c2766d09a0faa50fc1e4ece6793265f860ab752bb4c617`.
- Source function: `ACESFilm(float3 x)` with five-coefficient rational fit and saturation.
- License: the author explicitly offers the HLSL code under CC0 or MIT. This port chooses CC0-1.0; the exact permission statement, attribution, and source link are retained in `THIRD_PARTY_LICENSES/references/aces_fitted/NOTICE.txt`.
- Local upstream snapshot (ignored): `third_party/reference_sources/aces_fitted/aces-filmic-tone-mapping-curve.html`.
- Retrieved: 2026-10-08.

## Transform contract and fidelity

The original code constants are retained: a=2.51, b=0.03, c=2.43, d=0.59, e=0.14, followed by [0,1] saturation. It takes linear Rec.709 RGB, as clarified by the author in the same page's 2017-06-21 comment, and produces linear Rec.709 for later display encoding. The curve already includes the author's fitting exposure shift. The optional 0.6 input multiplier mentioned for approximating the unshifted ACES data is not applied; this is the published `ACESFilm` function as written.

The workbench adapter converts AP0/D60 to Rec.709/D65 and clips negative channels before the fit. Output encoding is delegated to the common wrapper. This is an SDR reference and ignores `headroom`. It is a luminance/curve approximation applied per RGB channel, not an ACES RRT/ODT or ACES 2 Output Transform; this distinction is preserved in its name and registry entry.

Independent double-precision fixtures include black, 18% gray, diffuse white, fit saturation and RGB highlights in `references/tests/simple_reference_vectors.json`; the generator is `references/tests/simple_reference_vectors.py`.

GPU fixtures also record `input_ap0` after rgba16float quantization and the reconstructed Rec.709 RGB. The primary expected output evaluates the independent author mathematics on that reconstructed input; unquantized expected values remain as metadata. This separates shader-port error from the input texture precision.
