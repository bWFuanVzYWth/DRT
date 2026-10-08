// Workbench adapter for Troy Sobotka's Filmic Log + Base Contrast preset.
// Original assets at 84bf836a4d9e130045c962c47ac4206395d4393b.
// Uses the complete 65^3 gamut grid and original 4096-sample contrast curve.
// Source/asset notices and input/output contract: references/entries/filmic.md.
fn reference_transform(ap0: vec3f, headroom: f32) -> vec3f {
    // Pinned OCIO Bradford AP0 -> Rec.709 conversion. Preserve its exact
    // adapter: tiny primary-edge differences are amplified by the log LUT.
    let rgb = vec3f(
        dot(ap0, vec3f(2.52168607711792, -1.1341309547424316, -0.38755524158477783)),
        dot(ap0, vec3f(-0.2764798700809479, 1.3727190494537354, -0.0962391272187233)),
        dot(ap0, vec3f(-0.015378059819340706, -0.15297536551952362, 1.1683534383773804)));
    let allocated = clamp((log2(max(rgb, vec3f(1e-20))) + 12.473931188) / 25.0,
        vec3f(0.0), vec3f(1.0));
    let filmic_log = reference_sample_cube(allocated, u32(reference_data[0]), u32(reference_data[1])) / 0.66;
    let offset = u32(reference_data[2]);
    let size = u32(reference_data[3]);
    let encoded = vec3f(reference_sample_curve(filmic_log.r, offset, size),
        reference_sample_curve(filmic_log.g, offset, size),
        reference_sample_curve(filmic_log.b, offset, size));
    return reference_srgb_decode(encoded);
}
