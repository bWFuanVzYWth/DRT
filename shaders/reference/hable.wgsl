// SPDX-License-Identifier: MIT
// Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
// Hable's Uncharted 2 operator from AMD Cauldron, ported to WGSL in 2026.
// Pinned source and adapter contract: references/entries/hable.md.

fn hable_uncharted2_curve(value: vec3f) -> vec3f {
    let a = 0.15;
    let b = 0.50;
    let c = 0.10;
    let d = 0.20;
    let e = 0.02;
    let f = 0.30;
    return (value * (a * value + vec3f(c * b)) + vec3f(d * e))
        / (value * (a * value + vec3f(b)) + vec3f(d * f)) - vec3f(e / f);
}

fn reference_transform(ap0: vec3f, headroom: f32) -> vec3f {
    let color = max(reference_ap0_to_rec709(ap0), vec3f(0.0));
    return hable_uncharted2_curve(2.0 * color) / hable_uncharted2_curve(vec3f(11.2));
}
