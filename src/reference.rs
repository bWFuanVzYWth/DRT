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
        DrtKind::Aces20Curve => include_str!("../shaders/research/aces_20_curve.wgsl"),
        DrtKind::BlenderAgx => include_str!("../shaders/reference/blender_agx.wgsl"),
        DrtKind::Filmic => include_str!("../shaders/reference/filmic.wgsl"),
        DrtKind::FidelityFxLpm => include_str!("../shaders/reference/fidelityfx_lpm.wgsl"),
        DrtKind::Hable => include_str!("../shaders/reference/hable.wgsl"),
        DrtKind::PbrNeutral => include_str!("../shaders/reference/pbr_neutral.wgsl"),
        DrtKind::Lottes => include_str!("../shaders/reference/lottes.wgsl"),
        DrtKind::OpenDrt => include_str!("../shaders/reference/opendrt.wgsl"),
        DrtKind::Uchimura => include_str!("../shaders/reference/uchimura.wgsl"),
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

pub fn data(kind: DrtKind, headroom: f32) -> Vec<f32> {
    match kind {
        DrtKind::Aces20 | DrtKind::Aces20Curve => crate::aces2_data::generate(headroom),
        DrtKind::FidelityFxLpm => crate::lpm_data::generate(headroom),
        DrtKind::BlenderAgx => floats(include_bytes!(
            "../shaders/reference/assets/blender_agx.bin"
        )),
        DrtKind::Filmic => floats(include_bytes!("../shaders/reference/assets/filmic.bin")),
        _ => vec![0.0; 4],
    }
}
