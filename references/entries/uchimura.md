# Uchimura / Gran Turismo Tone Mapping

- Shader: `shaders/reference/uchimura.wgsl`.
- Algorithm author: Hajime Uchimura, Polyphony Digital Inc. Course coauthor: Kentaro Suzuki.
- Official course page: https://www.polyphony.co.jp/publications/sa2018/
- Official course-page snapshot SHA-256: `b5d2345cc6ebca05f2bc9fa3991100fd1434436ca86641ce5da37c6bb6e402e1`.
- Official SIGGRAPH Asia 2018 slides: https://cdn2.gran-turismo.com/data/www/pdi_publications/PracticalHDRandWCGinGTS_20181222.pdf
- PDF snapshot SHA-256: `c21dbc6cf5d61eef3b1cc61b25d3492d728279ec2dbc9e28bf66ae4559236dbf` (13,651,732 bytes; 314 pages). The design is on pages 118-125; the analytic form is shown on page 184.
- Author's complete mathematical curve, directly linked from official slide page 124: https://www.desmos.com/calculator/mbkwnuihbd
- Author graph version: `mbkwnuihbd`, created 2017-08-24T17:36:01.177Z; graph identifies copyright 2017 Hajime Uchimura / Polyphony Digital.
- Exact graph-state source: https://www.desmos.com/calc-states/production/mbkwnuihbd
- Graph-state SHA-256: `38676ed7b78a0dc133e6c93593ecdfce5d663d23f7398f5ae18d7dec2928c0dc`.
- Local upstream files (ignored): `third_party/reference_sources/uchimura/` contains the course page, original PDF, author graph response, raw graph state, extraction and selected page renders.
- Retrieved: 2026-10-08.

## Implementation and rights

This repository independently implements the published mathematical formulas under the project's GPL-3.0-only license. No blanket MIT/Apache license is inferred for Polyphony's slides, text or original source snippets. The downloaded course material remains in the ignored local reference cache; it is not redistributed in Git. The attribution/implementation statement is retained in `THIRD_PARTY_LICENSES/references/uchimura/NOTICE.txt`.

## Transform contract and fidelity

The author graph's original SDR parameters are P=1, a=1, m=0.22, l=0.4, c=1.33, b=0. They are retained, except that the genuine author display-peak parameter P is supplied from `max(headroom,1)`. This directly supports SDR and HDR with the same published curve; it does not scale the completed SDR image. The original linear segment, power toe and exponential shoulder are evaluated analytically. The exponential's C2/P term is algebraically simplified to `a/(P-S1)`.

The workbench converts AP0/D60 to Rec.709/D65, clips negative channels, and applies the scalar curve independently to RGB. Output is display-linear Rec.709 relative to SDR white 1, approaching the chosen display peak. This reference implements the published tone curve only; the separate Gran Turismo gamut mapping and grading described elsewhere in the course are not claimed. The original game used a sampled LUT for runtime speed; the workbench uses the mathematically defined analytic curve.

Independent double-precision fixtures are generated directly from the author graph's intermediate C2/P definitions at P=1 and P=4. They cover the 0.22 linear start, 0.532 SDR shoulder join, black, saturated highlights and peak asymptotes. See `references/tests/simple_reference_vectors.json` and its `.py` generator.

GPU fixtures also record `input_ap0` after rgba16float quantization and the reconstructed Rec.709 RGB. The primary expected output evaluates the independent author mathematics on that reconstructed input; unquantized expected values remain as metadata. This separates shader-port error from the input texture precision.
