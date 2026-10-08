// SPDX-License-Identifier: MIT
// Copyright (c) 2017-2019 Advanced Micro Devices, Inc. All rights reserved.
// FidelityFX LPM 1.20200225; AMD, MIT license.
// Original: ffx_lpm.h at ed6ecd5b8963d2ec24603809b90ecfa00a1c3614.
// Specialization of LpmFilter/LpmMap: Rec2020 working -> Rec709 output,
// CON=true, SOFT=true, CON2=false, shoulderContrast=1, output peak scaling.
// CPU setup preserves the upstream first ten float4 control blocks.
// See references/entries/fidelityfx_lpm.md and src/lpm_data.rs.

fn lpm_sat(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }

fn lpm_soft(ratio: f32, gap: vec2<f32>) -> f32 {
    return min(max(gap.x, lpm_sat(ratio * -gap.x + ratio)),
        lpm_sat(gap.x * exp2(ratio * gap.y)));
}

fn reference_transform(ap0: vec3<f32>, headroom: f32) -> vec3<f32> {
    // AP0 D60 -> Rec2020 D65 using the workbench's chromatic adaptation.
    // LPM requires nonnegative working RGB. Clip after gamut conversion,
    // before taking ratios/powers, as required by its input contract.
    let rec709 = reference_ap0_to_rec709(ap0);
    let color = max(vec3<f32>(0.0), vec3<f32>(
        dot(rec709, vec3<f32>(0.627403895934699, 0.329283038377884, 0.043313065687417)),
        dot(rec709, vec3<f32>(0.069097289358232, 0.919540395075458, 0.011362315566310)),
        dot(rec709, vec3<f32>(0.016391438875150, 0.088013307877226, 0.895595253247624))));
    let max_color = max(color.r, max(color.g, color.b));
    // The upstream has undefined reciprocals at black; its continuous
    // limit is black. Avoid introducing NaNs into diagnostic plots.
    if (max_color == 0.0) { return vec3<f32>(0.0); }
    let saturation = vec3<f32>(reference_data[0], reference_data[1], reference_data[2]);
    let contrast = reference_data[3];
    let tone_bias = vec2<f32>(reference_data[4], reference_data[5]);
    let luma_t = vec3<f32>(reference_data[6], reference_data[7], reference_data[8]);
    let crosstalk = vec3<f32>(reference_data[9], reference_data[10], reference_data[11]);
    let rcp_luma_t = vec3<f32>(reference_data[12], reference_data[13], reference_data[14]);
    let luma_w = vec3<f32>(reference_data[25], reference_data[26], reference_data[27]);
    let gap = vec2<f32>(reference_data[28], reference_data[29]);
    let con_r = vec3<f32>(reference_data[30], reference_data[31], reference_data[32]);
    let con_g = vec3<f32>(reference_data[33], reference_data[34], reference_data[35]);
    let con_b = vec3<f32>(reference_data[36], reference_data[37], reference_data[38]);

    var ratio = pow(color / max_color, saturation);
    // Keep original operation order from LpmMap.
    var luma = color.g * luma_w.g + (color.r * luma_w.r + color.b * luma_w.b);
    luma = pow(luma, contrast);
    luma /= luma * tone_bias.x + tone_bias.y;
    let old_ratio = ratio;
    ratio = vec3<f32>(
        old_ratio.r * con_r.r + (old_ratio.g * con_r.g + old_ratio.b * con_r.b),
        old_ratio.g * con_g.g + (old_ratio.r * con_g.r + old_ratio.b * con_g.b),
        old_ratio.b * con_b.b + (old_ratio.g * con_b.g + old_ratio.r * con_b.r));
    ratio /= max(ratio.r, max(ratio.g, ratio.b));
    ratio = vec3<f32>(lpm_soft(ratio.r, gap), lpm_soft(ratio.g, gap), lpm_soft(ratio.b, gap));
    let luma_ratio = ratio.r * luma_t.r + ratio.g * luma_t.g + ratio.b * luma_t.b;
    let ratio_scale = lpm_sat(luma / luma_ratio);
    var mapped = clamp(ratio * ratio_scale, vec3<f32>(0.0), vec3<f32>(1.0));
    let cap = -crosstalk * mapped + crosstalk;
    var luma_add = lpm_sat(-mapped.b * luma_t.b + (-mapped.r * luma_t.r + (-mapped.g * luma_t.g + luma)));
    // At all-white the upstream has 0/0; the continuous limit is white.
    let denom = cap.g * luma_t.g + (cap.r * luma_t.r + cap.b * luma_t.b);
    if (denom > 0.0) {
        let t = luma_add / denom;
        mapped = clamp(t * cap + mapped, vec3<f32>(0.0), vec3<f32>(1.0));
    }
    luma_add = lpm_sat(-mapped.b * luma_t.b + (-mapped.r * luma_t.r + (-mapped.g * luma_t.g + luma)));
    mapped = clamp(luma_add * rcp_luma_t + mapped, vec3<f32>(0.0), vec3<f32>(1.0));
    // Identical to the upstream scaleOnly / scRGB container scaling step.
    return mapped * max(headroom, 1.0);
}
