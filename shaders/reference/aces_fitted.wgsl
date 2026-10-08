// SPDX-License-Identifier: CC0-1.0
// Krzysztof Narkowicz, ACESFilm (2016), ported to WGSL in 2026.
// This is the author's five-coefficient fit, not a full ACES transform.
// Pinned source and license grant: references/entries/aces_fitted.md.

fn reference_transform(ap0: vec3f, headroom: f32) -> vec3f {
    let x = max(reference_ap0_to_rec709(ap0), vec3f(0.0));
    let numerator = x * (2.51 * x + vec3f(0.03));
    let denominator = x * (2.43 * x + vec3f(0.59)) + vec3f(0.14);
    return clamp(numerator / denominator, vec3f(0.0), vec3f(1.0));
}
