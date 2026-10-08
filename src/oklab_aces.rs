//! An original Oklab experiment inspired by ACES 2's perceptual tone stage.
//! No ACES coefficients, white adaptation, or target-gamut boundary mapping.

pub const DEFAULT_HIGHLIGHT_REACH_EV: f32 = 10.0;
const MIN_SHOULDER_POWER: f32 = 0.25;
const MAX_SHOULDER_POWER: f32 = 8.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OklabAcesParameters {
    pub linear_slope: f32,
    pub compression_start: f32,
    pub shoulder_power: f32,
}

impl Default for OklabAcesParameters {
    fn default() -> Self {
        Self {
            linear_slope: 1.0,
            compression_start: 0.18,
            shoulder_power: power_for_highlight_reach(1.0, 0.18, 1.0, DEFAULT_HIGHLIGHT_REACH_EV),
        }
    }
}

impl OklabAcesParameters {
    pub fn maximum_compression_start(self) -> f32 {
        0.99 / self.linear_slope.clamp(0.1, 4.0)
    }

    pub fn constrain(&mut self) {
        let defaults = Self::default();
        self.linear_slope = finite_or(self.linear_slope, defaults.linear_slope).clamp(0.1, 4.0);
        self.compression_start = finite_or(self.compression_start, defaults.compression_start)
            .clamp(0.0, self.maximum_compression_start());
        self.shoulder_power = finite_or(self.shoulder_power, defaults.shoulder_power)
            .clamp(MIN_SHOULDER_POWER, MAX_SHOULDER_POWER);
    }

    pub fn curve_for_headroom(mut self, headroom: f32) -> OklabAcesCurve {
        self.constrain();
        OklabAcesCurve {
            linear_slope: self.linear_slope,
            compression_start: self.compression_start,
            output_peak: finite_or(headroom, 1.0).clamp(1.0, 64.0),
            shoulder_power: self.shoulder_power,
        }
    }

    /// Adjustable scene EV for 98% of the current display peak. The endpoints
    /// reflect the shoulder family, gain, join and actual display headroom.
    /// An equal range means 98% peak is already inside the linear segment.
    pub fn highlight_reach_range(self, headroom: f32) -> [f32; 2] {
        let mut curve = self.curve_for_headroom(headroom);
        curve.shoulder_power = MAX_SHOULDER_POWER;
        let minimum = curve.highlight_reach_ev();
        curve.shoulder_power = MIN_SHOULDER_POWER;
        // Use the complete physical range of the stored shape. A narrower
        // arbitrary EV cap could make the UI clamp a valid saved curve when
        // switching from SDR to a larger HDR peak.
        let maximum = curve.highlight_reach_ev().max(minimum);
        [minimum, maximum]
    }

    /// Change only the shoulder shape. The linear gain and join are retained.
    /// Store the resulting power so switching display headroom does not
    /// silently change the user's chosen curve.
    pub fn set_highlight_reach_ev(&mut self, reach_ev: f32, headroom: f32) {
        self.constrain();
        let curve = self.curve_for_headroom(headroom);
        let [minimum, maximum] = self.highlight_reach_range(curve.output_peak);
        if maximum - minimum <= 1.0e-5 {
            return;
        }
        let reach_ev = finite_or(reach_ev, curve.highlight_reach_ev()).clamp(minimum, maximum);
        self.shoulder_power = power_for_highlight_reach(
            self.linear_slope,
            self.compression_start,
            curve.output_peak,
            reach_ev,
        );
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OklabAcesCurve {
    pub linear_slope: f32,
    pub compression_start: f32,
    pub output_peak: f32,
    pub shoulder_power: f32,
}

impl OklabAcesCurve {
    /// x = Oklab L^3, which scales linearly with scene exposure.
    /// For x <= j, T(x) = k*x exactly. Above j, a generalized
    /// Michaelis-Menten shoulder is joined with the same first derivative:
    /// T = k*j + A*[1-(1+q/p)^(-p)], q = k*(x-j)/A, A = peak-k*j.
    /// p=1 gives the ordinary rational shoulder; p<1 extends its tail.
    /// Powers below 1 extend the tail without a finite input white point.
    pub fn map_linear(self, value: f32) -> f32 {
        if value <= self.compression_start {
            return self.linear_slope * value;
        }
        let output_join = self.linear_slope * self.compression_start;
        let extent = self.output_peak - output_join;
        let q = self.linear_slope * (value - self.compression_start) / extent;
        let progress = shoulder_progress(q, self.shoulder_power);
        output_join + extent * progress
    }

    /// Scene EV above 18% gray that reaches 98% of the configured peak.
    /// This is a useful length measurement, not a clipping point.
    pub fn highlight_reach_ev(self) -> f32 {
        let output_join = f64::from(self.linear_slope * self.compression_start);
        let peak = f64::from(self.output_peak);
        let target = 0.98 * peak;
        let input = if target <= output_join {
            target / f64::from(self.linear_slope)
        } else {
            let extent = peak - output_join;
            let remaining = 0.02 * peak / extent;
            let power = f64::from(self.shoulder_power);
            let q = power * (remaining.powf(-power.recip()) - 1.0);
            f64::from(self.compression_start) + extent * q / f64::from(self.linear_slope)
        };
        (input / 0.18).log2() as f32
    }
}

fn power_for_highlight_reach(gain: f32, join: f32, peak: f32, reach_ev: f32) -> f32 {
    // Construct bare curves here: Default uses this solver, so calling
    // Parameters::curve_for_headroom/constrain would recursively call Default.
    let curve = |power| OklabAcesCurve {
        linear_slope: gain,
        compression_start: join,
        output_peak: peak,
        shoulder_power: power,
    };
    let mut lower = MIN_SHOULDER_POWER;
    let mut upper = MAX_SHOULDER_POWER;
    let reach_ev = reach_ev.clamp(
        curve(upper).highlight_reach_ev(),
        curve(lower).highlight_reach_ev(),
    );
    for _ in 0..32 {
        let middle = (lower + upper) * 0.5;
        if curve(middle).highlight_reach_ev() > reach_ev {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    (lower + upper) * 0.5
}

fn finite_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() { value } else { fallback }
}

fn shoulder_progress(q: f32, power: f32) -> f32 {
    // Avoid subtracting nearly equal floats at the linear join. This cubic
    // expansion's omitted term is below 3e-11 over the parameter range.
    if q < 0.001 {
        q * (1.0 - (power + 1.0) / (2.0 * power) * q
            + (power + 1.0) * (power + 2.0) / (6.0 * power * power) * q * q)
    } else {
        1.0 - (1.0 + q / power).powf(-power)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shadows_are_linear_and_join_has_matching_value_and_slope() {
        for gain in [0.1, 1.0, 4.0] {
            for join in [0.0, 0.02, 0.18, 0.99 / gain] {
                for power in [0.25, 0.5, 1.0, 2.0, 8.0] {
                    let curve = OklabAcesParameters {
                        linear_slope: gain,
                        compression_start: join,
                        shoulder_power: power,
                    }
                    .curve_for_headroom(1.0);
                    for fraction in [0.0, 0.01, 0.5, 1.0] {
                        let input = join * fraction;
                        assert_eq!(curve.map_linear(input), gain * input);
                    }
                    // Analytic derivative: k*(1+q/p)^(-p-1), whose
                    // right limit is k. The right second derivative is
                    // finite for every allowed power as well.
                    let q = 1.0e-16_f64;
                    let right_slope = f64::from(gain)
                        * (1.0 + q / f64::from(power)).powf(-f64::from(power) - 1.0);
                    assert!((right_slope / f64::from(gain) - 1.0).abs() < 0.001);
                }
            }
        }
    }

    #[test]
    fn extended_highlights_are_monotone_bounded_and_reach_is_not_a_clip() {
        for headroom in [1.0, 4.0, 64.0] {
            for gain in [0.1, 1.0, 4.0] {
                for power in [0.25, 0.5, 1.0, 2.0, 8.0] {
                    let curve = OklabAcesParameters {
                        linear_slope: gain,
                        shoulder_power: power,
                        ..Default::default()
                    }
                    .curve_for_headroom(headroom);
                    let mut previous = 0.0;
                    for step in -160..=240 {
                        let input = 0.18 * 2.0_f32.powf(step as f32 / 8.0);
                        let output = curve.map_linear(input);
                        assert!(output.is_finite() && output >= previous);
                        assert!(output <= headroom && output <= gain * input + 1.0e-6);
                        previous = output;
                    }
                    let at_reach = 0.18 * 2.0_f32.powf(curve.highlight_reach_ev());
                    assert!((curve.map_linear(at_reach) / headroom - 0.98).abs() < 2.0e-6);
                    assert!(curve.map_linear(at_reach * 2.0) > curve.map_linear(at_reach));
                    assert!(curve.map_linear(f32::MAX).is_finite());
                }
            }
        }
        let curve = OklabAcesParameters::default().curve_for_headroom(1.0);
        assert!((curve.highlight_reach_ev() - DEFAULT_HIGHLIGHT_REACH_EV).abs() < 0.001);
        for ev in [6.5, 10.0, 14.0, 20.0] {
            assert!(curve.map_linear(0.18 * 2.0_f32.powf(ev)) < 1.0);
        }
    }

    #[test]
    fn highlight_reach_control_tracks_current_peak_and_preserves_shadows() {
        for gain in [0.1, 1.0, 4.0] {
            for join in [0.0, 0.18, 0.95 / gain] {
                for peak in [1.0, 4.0, 64.0] {
                    let original = OklabAcesParameters {
                        linear_slope: gain,
                        compression_start: join,
                        ..Default::default()
                    };
                    let [minimum, maximum] = original.highlight_reach_range(peak);
                    let mut previous_output = peak;
                    for reach in [minimum, (minimum + maximum) * 0.5, maximum] {
                        let mut edited = original;
                        edited.set_highlight_reach_ev(reach, peak);
                        let curve = edited.curve_for_headroom(peak);
                        assert!((curve.highlight_reach_ev() - reach).abs() < 1.0e-5);
                        let at_reach = 0.18 * 2.0_f32.powf(reach);
                        assert!((curve.map_linear(at_reach) / peak - 0.98).abs() < 2.0e-6);
                        assert_eq!(curve.map_linear(join * 0.5), gain * join * 0.5);
                        let output = curve.map_linear(join + peak / gain);
                        assert!(output <= previous_output + 1.0e-6);
                        previous_output = output;
                    }
                }
            }
        }
        // If 98% peak belongs to the linear segment, changing the shoulder
        // cannot move it. Keep the stored shape and expose a disabled range.
        let mut linear_target = OklabAcesParameters {
            compression_start: 0.99,
            ..Default::default()
        };
        let previous = linear_target.shoulder_power;
        let range = linear_target.highlight_reach_range(1.0);
        assert_eq!(range[0], range[1]);
        linear_target.set_highlight_reach_ev(8.0, 1.0);
        assert_eq!(linear_target.shoulder_power, previous);
    }

    #[test]
    fn constrain_keeps_usable_parameters_for_invalid_values() {
        let mut parameters = OklabAcesParameters {
            linear_slope: f32::NAN,
            compression_start: f32::INFINITY,
            shoulder_power: -1.0,
        };
        parameters.constrain();
        assert_eq!(parameters.linear_slope, 1.0);
        assert_eq!(parameters.compression_start, 0.18);
        assert_eq!(parameters.shoulder_power, 0.25);
        assert_eq!(parameters.curve_for_headroom(f32::NAN).output_peak, 1.0);
    }
}
