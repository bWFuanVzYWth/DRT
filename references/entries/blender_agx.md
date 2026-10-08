# Blender official AgX: Base Rec.1886, without a look

- Official remote: https://github.com/blender/blender
- Immutable commit: `8b5bd560cf16070d10f361d0a9f00b39436dc379`.
- [Pinned config.ocio](https://github.com/blender/blender/blob/8b5bd560cf16070d10f361d0a9f00b39436dc379/release/datafiles/colormanagement/config.ocio).
- [Pinned AgX_Base_sRGB.cube](https://github.com/blender/blender/blob/8b5bd560cf16070d10f361d0a9f00b39436dc379/release/datafiles/colormanagement/luts/AgX_Base_sRGB.cube).
- Authors credited by config: original AgX by Troy Sobotka; further development by Zijun Eary Zhou, Mark Faderbauer and Sakari Kapanen. The config links https://github.com/EaryChow/AgX for this version.
- Local original tree: `third_party/reference_sources/blender/` (sparse checkout, ignored).
- Port: `shaders/reference/blender_agx.wgsl`; packed runtime table: `shaders/reference/assets/blender_agx.bin`.
- Source SHA256 and matrix provenance: `references/validation/lut_assets.json`.
- License/asset notice: `THIRD_PARTY_LICENSES/references/blender_agx/NOTICE.md`. The pinned config refers to `ocio-license.txt`, which is absent from this checkout. The asset license is recorded as **NOASSERTION**, not guessed from another implementation or claimed as MIT.
- Retrieved and ported: 2026-10-08.

The reference preset is Blender's official **AgX Base Rec.1886** view rendered to an sRGB display, with no contrast/color look. This is a separate algorithm version from AgX-S2O3. The complete official 57^3 grid is retained as little-endian float32 RGB triplets and sampled with OCIO-style tetrahedral interpolation; no polynomial approximation replaces it.

Input: scene-linear AP0. The pinned config's AP0 -> XYZ D65 Bradford -> FilmLight E-Gamut matrix is reproduced. Log allocation is `[-12.47393, 12.5260688117]`. The grid encodes Rec.1886 values, so the port applies gamma 2.4 to obtain display-linear Rec.709. The common adapter performs the workbench's final sRGB/scRGB encoding.

Output: SDR reference preset, white = 1. The source tree also includes Blender's 1000-nit HLG/P3-limited grid for local study, but this selector deliberately represents the stated SDR preset; it is not a rescaled HDR approximation. HDR headroom changes do not alter it.

Reproduce packing and independent vectors with `python references/validation/generate_lut_assets.py`; `--check` verifies byte-for-byte regeneration. Requires NumPy and PyOpenColorIO 2.6.0 in the ignored local validation environment. The generator records the original grid SHA256, source commit and composite matrix. GPU checks compare against the official Blender OCIO display view on 33 half-quantized AP0 samples covering black, gray, color primaries, saturated highlights, negative inputs and deep shadows.
