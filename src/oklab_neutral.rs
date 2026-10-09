//! Max-RGB parameters for the root-LMS neutral experiment.
//! Shares the scalar shoulder family with Oklab ACES-inspired, while keeping
//! independent defaults and state. The color path is entirely in its WGSL.

use crate::oklab_aces::{OklabAcesCurve, OklabAcesParameters};

pub const DEFAULT_COMPRESSION_START: f32 = 0.6;
pub const DEFAULT_HIGHLIGHT_REACH_EV: f32 = 10.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OklabNeutralParameters {
    pub linear_slope: f32,
    pub compression_start: f32,
    pub shoulder_power: f32,
}

impl Default for OklabNeutralParameters {
    fn default() -> Self {
        let mut tone = OklabAcesParameters {
            compression_start: DEFAULT_COMPRESSION_START,
            ..OklabAcesParameters::default()
        };
        tone.set_highlight_reach_ev(DEFAULT_HIGHLIGHT_REACH_EV, 1.0);
        Self::from_tone(tone)
    }
}

impl OklabNeutralParameters {
    fn tone(self) -> OklabAcesParameters {
        OklabAcesParameters {
            linear_slope: self.linear_slope,
            compression_start: self.compression_start,
            shoulder_power: self.shoulder_power,
        }
    }

    fn from_tone(tone: OklabAcesParameters) -> Self {
        Self {
            linear_slope: tone.linear_slope,
            compression_start: tone.compression_start,
            shoulder_power: tone.shoulder_power,
        }
    }

    pub fn constrain(&mut self) {
        let defaults = Self::default();
        if !self.linear_slope.is_finite() {
            self.linear_slope = defaults.linear_slope;
        }
        if !self.compression_start.is_finite() {
            self.compression_start = defaults.compression_start;
        }
        if !self.shoulder_power.is_finite() {
            self.shoulder_power = defaults.shoulder_power;
        }
        let mut tone = self.tone();
        tone.constrain();
        *self = Self::from_tone(tone);
    }

    pub fn maximum_compression_start(self) -> f32 {
        self.tone().maximum_compression_start()
    }

    /// The scalar input is max(linear Rec.709 RGB), rather than Oklab L^3.
    pub fn curve_for_headroom(mut self, headroom: f32) -> OklabAcesCurve {
        self.constrain();
        self.tone().curve_for_headroom(headroom)
    }

    pub fn highlight_reach_range(mut self, headroom: f32) -> [f32; 2] {
        self.constrain();
        self.tone().highlight_reach_range(headroom)
    }

    pub fn set_highlight_reach_ev(&mut self, reach_ev: f32, headroom: f32) {
        self.constrain();
        let mut tone = self.tone();
        tone.set_highlight_reach_ev(reach_ev, headroom);
        *self = Self::from_tone(tone);
    }
}
