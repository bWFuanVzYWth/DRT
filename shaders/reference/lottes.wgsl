// SPDX-License-Identifier: MIT
// Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
// AMD Cauldron implementation of Timothy Lottes' max-RGB/crosstalk operator.
// Ported to WGSL in 2026. Pinned source: references/entries/lottes.md.

fn lottes_anchor_coefficients() -> vec2f {
    let hdr_max = 16.0;
    let contrast = 2.0;
    let shoulder = 1.0;
    let middle_input = 0.18;
    let middle_output = 0.18;
    let high_contrast = pow(hdr_max, contrast);
    let high_shoulder = pow(hdr_max, contrast * shoulder);
    let middle_contrast = pow(middle_input, contrast);
    let middle_shoulder = pow(middle_input, contrast * shoulder);
    let denominator = (high_shoulder - middle_shoulder) * middle_output;
    // Algebraically equivalent to upstream ColToneB/ColToneC, avoiding
    // the nested subtraction in the original ColToneB expression.
    let b = (high_contrast * middle_output - middle_contrast) / denominator;
    let c = (high_shoulder * middle_contrast
        - high_contrast * middle_shoulder * middle_output) / denominator;
    return vec2f(b, c);
}

fn reference_transform(ap0: vec3f, headroom: f32) -> vec3f {
    let color = max(reference_ap0_to_rec709(ap0), vec3f(0.0));
    let contrast = 2.0;
    let shoulder = 1.0;
    let coefficients = lottes_anchor_coefficients();
    let input_peak = max(1.0e-6, max(color.r, max(color.g, color.b)));
    let z = pow(input_peak, contrast);
    let output_peak = z / (pow(z, shoulder) * coefficients.x + coefficients.y);
    let crosstalk = 4.0;
    let saturation = contrast;
    let cross_saturation = contrast * 16.0;
    var ratio = pow(abs(color / input_peak), vec3f(saturation / cross_saturation));
    ratio = mix(ratio, vec3f(1.0), pow(output_peak, crosstalk));
    ratio = pow(abs(ratio), vec3f(cross_saturation));
    return output_peak * ratio;
}
