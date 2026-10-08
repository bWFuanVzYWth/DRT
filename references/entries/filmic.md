# Filmic: original gamut grid and Base Contrast look

- Original author: Troy James Sobotka; additional contributors are credited in the pinned config.
- Remote: https://github.com/sobotka/filmic-blender
- Immutable commit: `84bf836a4d9e130045c962c47ac4206395d4393b`.
- [Pinned config](https://github.com/sobotka/filmic-blender/blob/84bf836a4d9e130045c962c47ac4206395d4393b/config.ocio).
- [Original 65^3 gamut LUT](https://github.com/sobotka/filmic-blender/blob/84bf836a4d9e130045c962c47ac4206395d4393b/luts/desat65cube.spi3d).
- [Original Base Contrast curve](https://github.com/sobotka/filmic-blender/blob/84bf836a4d9e130045c962c47ac4206395d4393b/looks/Filmic_to_0-70_1-03.spi1d).
- Local checkout: `third_party/reference_sources/filmic/` (ignored).
- Port: `shaders/reference/filmic.wgsl`; runtime table: `shaders/reference/assets/filmic.bin`.
- Original asset SHA256 values: `references/validation/lut_assets.json`.
- License notice: `THIRD_PARTY_LICENSES/references/filmic/NOTICE.md`. The pinned repository does not contain a standalone license granting a specific MIT/BSD/CC0 license; original assets are recorded as **NOASSERTION** and their attribution is retained.
- Retrieved and ported: 2026-10-08.

The explicitly selected reference is **Filmic Log + Base Contrast**, not all possible Filmic looks. The full original 65^3 gamut grid and 4096-entry Base Contrast curve are preserved. The workbench adds an AP0 input adapter using the pinned Blender OCIO Bradford AP0 -> linear Rec.709 matrix. This exact matrix matters near saturated primary edges: using the workbench's slightly different generic conversion creates small negative/positive components that the log gamut grid can amplify.

Pipeline: AP0 -> linear Rec.709 -> log2 allocation `[-12.473931188, 12.526068812]` -> original gamut LUT with tetrahedral interpolation -> uniform allocation `[0, 0.66]` -> original Base Contrast LUT with linear interpolation -> sRGB decoding. Result: display-linear SDR Rec.709, white = 1. Final encoding and anomaly visualization remain in the shared workbench adapter. No ad hoc HDR extension or polynomial curve replaces the LUTs.

`references/validation/generate_lut_assets.py` deterministically packs the original tables and generates an independent PyOpenColorIO 2.6.0 CPU GroupTransform using the same source assets. It records 33 half-quantized AP0 samples and all input/output conventions in `lut_reference_vectors.json`; GPU checks cover primaries, saturated highlights, gray levels, negative values and deep shadows. The runtime does not depend on Python, OCIO or the ignored checkouts.
