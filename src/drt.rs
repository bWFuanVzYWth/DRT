#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrtKind {
    None,
    AgxS2O3,
    Aces13,
    Aces20,
    BlenderAgx,
    Filmic,
    FidelityFxLpm,
    Hable,
    PbrNeutral,
    Lottes,
    OpenDrt,
    Uchimura,
    AcesFitted,
    OklabLogSigmoid,
    OklabReinhard,
    OklabAces,
    OklabNeutral,
    RgbLogSigmoid,
    RgbReinhard,
}

impl DrtKind {
    pub const REFERENCES: [Self; 12] = [
        Self::AgxS2O3,
        Self::Aces13,
        Self::Aces20,
        Self::BlenderAgx,
        Self::Filmic,
        Self::FidelityFxLpm,
        Self::Hable,
        Self::PbrNeutral,
        Self::Lottes,
        Self::OpenDrt,
        Self::Uchimura,
        Self::AcesFitted,
    ];
    pub const RESEARCH: [Self; 6] = [
        Self::OklabLogSigmoid,
        Self::OklabReinhard,
        Self::OklabAces,
        Self::OklabNeutral,
        Self::RgbLogSigmoid,
        Self::RgbReinhard,
    ];
    pub const ALL: [Self; 19] = [
        Self::None,
        Self::AgxS2O3,
        Self::Aces13,
        Self::Aces20,
        Self::BlenderAgx,
        Self::Filmic,
        Self::FidelityFxLpm,
        Self::Hable,
        Self::PbrNeutral,
        Self::Lottes,
        Self::OpenDrt,
        Self::Uchimura,
        Self::AcesFitted,
        Self::OklabLogSigmoid,
        Self::OklabReinhard,
        Self::OklabAces,
        Self::OklabNeutral,
        Self::RgbLogSigmoid,
        Self::RgbReinhard,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::None => "None",
            Self::AgxS2O3 => "AgX-S2O3",
            Self::Aces13 => "ACES 1.3 RRT + ODT",
            Self::Aces20 => "ACES 2.0",
            Self::BlenderAgx => "Blender AgX",
            Self::Filmic => "Filmic Base Contrast",
            Self::FidelityFxLpm => "AMD FidelityFX LPM",
            Self::Hable => "Hable / Uncharted 2",
            Self::PbrNeutral => "Khronos PBR Neutral",
            Self::Lottes => "Lottes 2016",
            Self::OpenDrt => "OpenDRT",
            Self::Uchimura => "Uchimura / GT",
            Self::AcesFitted => "ACES Filmic Fitted",
            Self::OklabLogSigmoid => "Oklab Log Sigmoid",
            Self::OklabReinhard => "Oklab Reinhard",
            Self::OklabAces => "Oklab ACES-inspired",
            Self::OklabNeutral => "Oklab Neutral",
            Self::RgbLogSigmoid => "RGB Log Sigmoid",
            Self::RgbReinhard => "RGB Reinhard",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::None => "Display conversion and clipping without tone compression",
            Self::AgxS2O3 => "linlin's analytic AgX-S2O3 reference port; SDR output",
            Self::Aces13 => {
                "Academy ACES 1.3 RRT and sRGB 100-nit dim-surround ODT; SDR reference preset"
            }
            Self::Aces20 => {
                "Academy ACES 2 JMh tonescale, chroma compression and gamut mapping; Rec.709 SDR/HDR"
            }
            Self::BlenderAgx => {
                "Blender's official E-Gamut AgX base LUT with tetrahedral interpolation; SDR Rec.1886 reference"
            }
            Self::Filmic => {
                "Troy Sobotka's Filmic gamut compression and Base Contrast look; SDR reference"
            }
            Self::FidelityFxLpm => {
                "AMD luminance-preserving tone and gamut mapping; Rec.2020 working gamut, Rec.709 SDR/HDR output"
            }
            Self::Hable => {
                "John Hable's Uncharted 2 filmic curve and original white-point normalization; SDR"
            }
            Self::PbrNeutral => {
                "Khronos PBR Neutral: preserve base colors and desaturate highlights; SDR"
            }
            Self::Lottes => {
                "Timothy Lottes's AMD max-RGB tone mapper and crosstalk; original SDR parameters"
            }
            Self::OpenDrt => {
                "Jed Smith's OpenDRT Standard look, tone and purity compression; Rec.709 SDR/HDR"
            }
            Self::Uchimura => {
                "Hajime Uchimura's Gran Turismo toe, linear midsection and shoulder; variable display peak"
            }
            Self::AcesFitted => {
                "Krzysztof Narkowicz's five-coefficient ACES Filmic fit; SDR approximation"
            }
            Self::OklabLogSigmoid => {
                "Analytic linear shadows and log2 sigmoid shoulder in Oklab L^3 with chroma compression"
            }
            Self::OklabReinhard => {
                "Linear segment and Reinhard shoulder in Oklab L^3 with chroma compression"
            }
            Self::OklabAces => {
                "Experimental ACES-inspired shoulder with fixed Oklab hue and saturation detail near white; linear tone shadows and SDR/HDR"
            }
            Self::OklabNeutral => {
                "Experimental max-RGB shoulder and root-LMS white path; linear Rec.709 shadows and SDR/HDR"
            }
            Self::RgbLogSigmoid => {
                "Research RGB log2 sigmoid with exact linear shadows and HSV hue repair"
            }
            Self::RgbReinhard => {
                "Research RGB linear segment and Reinhard shoulder with HSV hue repair"
            }
        }
    }

    pub fn shader_file(self) -> &'static str {
        match self {
            Self::None => "research/none_drt.wgsl",
            Self::AgxS2O3 => "reference/agx_s2o3.wgsl",
            Self::Aces13 => "reference/aces_13.wgsl",
            Self::Aces20 => "reference/aces_20.wgsl",
            Self::BlenderAgx => "reference/blender_agx.wgsl",
            Self::Filmic => "reference/filmic.wgsl",
            Self::FidelityFxLpm => "reference/fidelityfx_lpm.wgsl",
            Self::Hable => "reference/hable.wgsl",
            Self::PbrNeutral => "reference/pbr_neutral.wgsl",
            Self::Lottes => "reference/lottes.wgsl",
            Self::OpenDrt => "reference/opendrt.wgsl",
            Self::Uchimura => "reference/uchimura.wgsl",
            Self::AcesFitted => "reference/aces_fitted.wgsl",
            Self::OklabLogSigmoid => "research/oklab_log_sigmoid.wgsl",
            Self::OklabReinhard => "research/oklab_reinhard.wgsl",
            Self::OklabAces => "research/oklab_aces.wgsl",
            Self::OklabNeutral => "research/oklab_neutral.wgsl",
            Self::RgbLogSigmoid => "research/rgb_log_sigmoid.wgsl",
            Self::RgbReinhard => "research/rgb_reinhard.wgsl",
        }
    }

    pub fn index(self) -> usize {
        Self::ALL
            .iter()
            .position(|kind| *kind == self)
            .expect("registered DRT")
    }

    pub fn is_reference(self) -> bool {
        Self::REFERENCES.contains(&self)
    }
    pub fn uses_reference_wrapper(self) -> bool {
        self.is_reference() && self != Self::AgxS2O3
    }
    pub fn uses_linear_log_sigmoid(self) -> bool {
        matches!(self, Self::RgbLogSigmoid | Self::OklabLogSigmoid)
    }
    pub fn is_oklab(self) -> bool {
        matches!(
            self,
            Self::OklabReinhard | Self::OklabLogSigmoid | Self::OklabAces | Self::OklabNeutral
        )
    }
    pub fn supports_hdr(self) -> bool {
        matches!(
            self,
            Self::None
                | Self::RgbReinhard
                | Self::RgbLogSigmoid
                | Self::OklabAces
                | Self::OklabNeutral
                | Self::Aces20
                | Self::FidelityFxLpm
                | Self::OpenDrt
                | Self::Uchimura
        )
    }

    pub fn source_url(self) -> Option<&'static str> {
        Some(match self {
            Self::AgxS2O3 => {
                "https://github.com/bWFuanVzYWth/AgX/tree/0796e1b4aa9df94152eff353bae131eae1a4c087"
            }
            Self::Aces13 => {
                "https://github.com/aces-aswf/aces-core/tree/1256fee50ee35548c6eab8eca854ff3349008489"
            }
            Self::Aces20 => {
                "https://github.com/aces-aswf/aces-core/tree/069b0bc3e1f6c62820f19fdae2fecec3f4fc0f80"
            }
            Self::BlenderAgx => {
                "https://github.com/blender/blender/tree/8b5bd560cf16070d10f361d0a9f00b39436dc379/release/datafiles/colormanagement"
            }
            Self::Filmic => {
                "https://github.com/sobotka/filmic-blender/tree/84bf836a4d9e130045c962c47ac4206395d4393b"
            }
            Self::FidelityFxLpm => {
                "https://github.com/GPUOpen-Effects/FidelityFX-LPM/blob/ed6ecd5b8963d2ec24603809b90ecfa00a1c3614/ffx-lpm/ffx_lpm.h"
            }
            Self::Hable | Self::Lottes => {
                "https://github.com/GPUOpen-LibrariesAndSDKs/Cauldron/blob/b92d559bd083f44df9f8f42a6ad149c1584ae94c/src/VK/shaders/tonemappers.glsl"
            }
            Self::PbrNeutral => {
                "https://github.com/KhronosGroup/ToneMapping/blob/180b1a7bddec33f73fe41712a2963cc3ad8e5547/PBR_Neutral/pbrNeutral.glsl"
            }
            Self::OpenDrt => {
                "https://github.com/jedypod/open-display-transform/blob/af683323e2a8a63501f02c0a724ec538e3228ad0/display-transforms/opendrt/OpenDRT.dctl"
            }
            Self::Uchimura => "https://www.polyphony.co.jp/publications/sa2018/",
            Self::AcesFitted => {
                "https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/"
            }
            _ => return None,
        })
    }
}
