# AgX-S2O3 reference port

- Original analytic GLSL port author: linlin (2024).
- Remote repository: https://github.com/bWFuanVzYWth/AgX
- Immutable commit: `0796e1b4aa9df94152eff353bae131eae1a4c087`.
- Direct reference: [agx.glsl](https://github.com/bWFuanVzYWth/AgX/blob/0796e1b4aa9df94152eff353bae131eae1a4c087/agx.glsl).
- Original algorithm lineage: [Troy Sobotka's AgX](https://github.com/sobotka/AgX), as linked by the pinned GLSL repository's README.
- Local source checkout: `third_party/reference_sources/agx-s2o3/` (ignored by Git).
- Workbench port: `shaders/reference/agx_s2o3.wgsl`, moved from its previous location without changing the algorithm.
- Upstream license: MIT, copyright (c) 2024 linlin. Retained in `THIRD_PARTY_LICENSES/references/agx_s2o3/LICENSE`.
- Workbench derivative: GPL-3.0-only, consistent with the existing port header and project license.
- Retrieved and registered: 2026-10-08.

This is the existing analytic S2O3 port, not the newer Blender/E-Gamut LUT version. The input is scene-linear ACES2065-1/AP0, converted through the workbench's existing display conversion. Output is SDR extended-sRGB-encoded Rec.709. The separate source implementation is kept locally and its remote commit is now recorded instead of relying only on `C:/WorkSpace/AgX/agx.glsl`.

The workbench's existing AgX controls and reference defaults remain unchanged: inset 0.2; shadow reach -10 EV and highlight reach +6.5 EV relative to 18% gray; output pivot 0.5; original toe/shoulder powers and mid-contrast. Reset restores the existing reference preset. Exposure and anomaly coloring belong to the workbench adapter; HDR headroom does not turn this SDR reference into an HDR algorithm.

Validation preserves the existing scalar endpoint/pivot/tangent tests and actual GPU output, nonfinite diagnostics, neutral-axis, comparison and shader-reload tests. The shader was relocated, not replaced by a fitted or Blender LUT implementation.
