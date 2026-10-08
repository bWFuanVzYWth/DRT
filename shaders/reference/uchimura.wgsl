// SPDX-License-Identifier: GPL-3.0-only
// Independently implemented from the published Uchimura/GT mathematical
// curve, rather than copied slide/source implementation text.
// Formula provenance and output-peak adaptation: references/entries/uchimura.md.

fn uchimura_component(input: f32, display_peak: f32) -> f32 {
    let x = max(input, 0.0);
    let peak = max(display_peak, 1.0);
    let contrast = 1.0;
    let middle = 0.22;
    let linear_length = 0.40;
    let toe_power = 1.33;
    let black = 0.0;
    let linear_extent = (peak - middle) * linear_length / contrast;
    let shoulder_input = middle + linear_extent;
    let shoulder_output = middle + contrast * linear_extent;
    let shoulder_rate = contrast / (peak - shoulder_output);
    let toe_weight = 1.0 - smoothstep(0.0, middle, x);
    let shoulder_weight = select(0.0, 1.0, x >= shoulder_input);
    let linear_weight = 1.0 - toe_weight - shoulder_weight;
    let toe = middle * pow(x / middle, toe_power) + black;
    let linear = middle + contrast * (x - middle);
    let shoulder = peak - (peak - shoulder_output)
        * exp(-shoulder_rate * (x - shoulder_input));
    return toe * toe_weight + linear * linear_weight + shoulder * shoulder_weight;
}

fn reference_transform(ap0: vec3f, headroom: f32) -> vec3f {
    let color = reference_ap0_to_rec709(ap0);
    return vec3f(
        uchimura_component(color.r, headroom),
        uchimura_component(color.g, headroom),
        uchimura_component(color.b, headroom));
}
