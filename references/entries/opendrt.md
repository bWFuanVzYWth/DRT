# OpenDRT 1.1.0 — Standard

- Author: Jed Smith.
- Upstream repository: https://github.com/jedypod/open-display-transform
- Immutable commit: `af683323e2a8a63501f02c0a724ec538e3228ad0`.
- Reference: [OpenDRT.dctl](https://github.com/jedypod/open-display-transform/blob/af683323e2a8a63501f02c0a724ec538e3228ad0/display-transforms/opendrt/OpenDRT.dctl).
- Original SHA-256: `d937afee09519d04aff35bff789980b9718920c567fd5269e3888683231c664c`.
- Local upstream checkout: `third_party/reference_sources/opendrt/` (ignored by Git).
- Port: `shaders/reference/opendrt.wgsl`.
- License: GPL-3.0; upstream license is preserved in `THIRD_PARTY_LICENSES/references/opendrt/LICENSE`.
- Retrieved and ported: 2026-10-08.

The port specializes the complete original **Standard** look with scene-linear ACES2065-1/AP0 input, Rec.709 D65 output, D65 creative white and the Rec.1886 preset's **dim surround**. It retains the original AP0-to-XYZ CAT02 matrix, P3 rendering space, Euclidean tonescale norm, brilliance, red hue contrast, RGB/CMY hue shifts, low/high purity limits, midrange purity, post brilliance and purity soft clipping. Original input transfer decoding, optional diagnostic overlay and unused presets are omitted. Contrast Low and Contrast High are disabled by the original Standard preset, so their inactive branches are omitted.

Default tone settings are contrast 1.66, shoulder 0.5, toe 0.003, offset 0.005, grey luminance 10 cd/m², HDR grey boost 0.13 and HDR purity 0.5. All look settings remain the original Standard values, shown alongside their operations in the shader. Signed AP0 values are preserved through the original gamut matrices and rendering operations; this port does not add a pre-tonemap negative clamp.

Display peak is `100 * headroom` cd/m². The original Rec.1886 preset normalizes linear output by that peak, clips to [0,1], and applies power 1/2.4. The workbench adapter omits that encoding and removes only the peak normalization, returning **linear Rec.709 in units of 100 cd/m² SDR white**, clipped to [0,headroom]. Thus headroom 1 reproduces the SDR reference; higher peaks use the original HDR tonescale and purity controls with the same Standard look and Rec.709 gamut. This is a workbench linear-output adaptation of the author’s Rec.1886 preset, not the author's Rec.2100 PQ/P3-limited Rec.2020 preset. The workbench performs the final presentation encoding.

Validation compiles the actual pinned DCTL with Clang's native vector types and DCTL helper aliases; `transform()` is otherwise unmodified. Reference output is converted back from its power-2.4 encoding and multiplied by headroom. On the available GPU, 33 samples covering black, grey, neutral highlights, AP0 primaries, negative input and strong overexposure at headroom 1/4/10 differed by at most **4.02e-6** in display-linear output. Tracked upstream golden values and their regeneration script are in `references/validation/hdr_reference_vectors.json` and `generate_hdr_vectors.py`.
