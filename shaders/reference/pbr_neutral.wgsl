// SPDX-License-Identifier: Apache-2.0
// Copyright 2024 The Khronos Group Inc.
// Modified for WGSL and the workbench AP0 adapter, 2026.
// Pinned source and adapter contract: references/entries/pbr_neutral.md.

fn reference_transform(ap0: vec3f, headroom: f32) -> vec3f {
    var color = max(reference_ap0_to_rec709(ap0), vec3f(0.0));
    let start_compression = 0.8 - 0.04;
    let desaturation = 0.15;
    let minimum = min(color.r, min(color.g, color.b));
    let offset = select(0.04, minimum - 6.25 * minimum * minimum, minimum < 0.08);
    color -= vec3f(offset);
    let peak = max(color.r, max(color.g, color.b));
    if peak < start_compression {
        return color;
    }
    let distance = 1.0 - start_compression;
    let new_peak = 1.0 - distance * distance / (peak + distance - start_compression);
    color *= new_peak / peak;
    let whitening = 1.0 - 1.0 / (desaturation * (peak - new_peak) + 1.0);
    return mix(color, vec3f(new_peak), whitening);
}
