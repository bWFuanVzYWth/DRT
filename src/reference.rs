//! Reference shader composition and immutable/runtime reference data.
//! Each port has a pinned source record in references/entries/.
use crate::drt::DrtKind;

pub const COMMON: &str = include_str!("../shaders/reference/common.wgsl");
pub const ENTRY: &str = include_str!("../shaders/reference/entry.wgsl");

pub fn compose(fragment: &str) -> String {
    format!("{COMMON}\n{fragment}\n{ENTRY}")
}

pub fn fragment(kind: DrtKind) -> &'static str {
    match kind {
        DrtKind::Aces13 => include_str!("../shaders/reference/aces_13.wgsl"),
        DrtKind::Aces20 => include_str!("../shaders/reference/aces_20.wgsl"),
        DrtKind::BlenderAgx => include_str!("../shaders/reference/blender_agx.wgsl"),
        DrtKind::Filmic => include_str!("../shaders/reference/filmic.wgsl"),
        DrtKind::FidelityFxLpm => include_str!("../shaders/reference/fidelityfx_lpm.wgsl"),
        DrtKind::Hable => include_str!("../shaders/reference/hable.wgsl"),
        DrtKind::PbrNeutral => include_str!("../shaders/reference/pbr_neutral.wgsl"),
        DrtKind::Lottes => include_str!("../shaders/reference/lottes.wgsl"),
        DrtKind::OpenDrt => include_str!("../shaders/reference/opendrt.wgsl"),
        DrtKind::Uchimura => include_str!("../shaders/reference/uchimura.wgsl"),
        DrtKind::Gt7 => include_str!("../shaders/reference/gt7.wgsl"),
        DrtKind::AcesFitted => include_str!("../shaders/reference/aces_fitted.wgsl"),
        _ => panic!("DRT has no reference fragment"),
    }
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect()
}

// GT7 v1.0 initialization, Copyright (c) 2025 Polyphony Digital Inc. (MIT).
// Source/license and the 250-nit output-unit adapter: references/entries/gt7.md.
fn gt7_parameters(headroom: f32) -> Vec<f32> {
    let peak = 2.5 * headroom.clamp(1.0, 40.0);
    let alpha: f32 = 0.25;
    let linear_section: f32 = 0.444;
    let k = (linear_section - 1.0) / (alpha - 1.0);
    let k_a = peak * linear_section + peak * k;
    let k_b = -peak * k * (linear_section / k).exp();
    let k_c = -1.0 / (k * peak);
    // Official inverse ST-2084: framebuffer 1 is 100 nits.
    let y = peak * 100.0 / 10_000.0;
    let ym = y.powf(0.159_301_76);
    let target_i = (78.843_75
        * ((0.835_937_5 + 18.851_563 * ym).log2() - (1.0 + 18.687_5 * ym).log2()))
    .exp2();
    vec![peak, target_i, k_a, k_b, k_c, 0.4, 0.0, 0.0]
}

pub fn data(kind: DrtKind, headroom: f32) -> Vec<f32> {
    match kind {
        DrtKind::Gt7 => gt7_parameters(headroom),
        DrtKind::Aces20 => crate::aces2_data::generate(headroom),
        DrtKind::FidelityFxLpm => crate::lpm_data::generate(headroom),
        DrtKind::BlenderAgx => floats(include_bytes!(
            "../shaders/reference/assets/blender_agx.bin"
        )),
        DrtKind::Filmic => floats(include_bytes!("../shaders/reference/assets/filmic.bin")),
        _ => vec![0.0; 4],
    }
}
