// SPDX-License-Identifier: GPL-3.0-only
// Common presentation/diagnostic adapter; the reference algorithm returns
// display-linear Rec.709. Keep encoding and range clipping out of its math.
fn reference_has_nan(value: vec3f) -> bool {
    return any((bitcast<vec3u>(value) & vec3u(0x7fffffffu)) > vec3u(0x7f800000u));
}
fn reference_has_pos_inf(value: vec3f) -> bool {
    return any(bitcast<vec3u>(value) == vec3u(0x7f800000u));
}
fn reference_has_neg_inf(value: vec3f) -> bool {
    return any(bitcast<vec3u>(value) == vec3u(0xff800000u));
}
fn reference_exception_color(positive: bool, negative: bool) -> vec3f {
        if (parameters.showAnomalies != 0u) {
            if (positive && negative) { return vec3f(1.0, 0.35, 0.0); }
            return select(vec3f(0.0, 1.0, 1.0), vec3f(1.0, 1.0, 0.0), positive);
        }
        if (positive && negative) { return vec3f(1.0, 0.0, 1.0); }
        return select(vec3f(0.0), vec3f(1.0), positive);
}
fn reference_prepare_output(source: vec3f, mapped: vec3f) -> vec3f {
    if (reference_has_nan(source)) { return vec3f(1.0, 0.0, 1.0); }
    let source_positive = reference_has_pos_inf(source);
    let source_negative = reference_has_neg_inf(source);
    if (source_positive || source_negative) {
        return reference_exception_color(source_positive, source_negative);
    }
    if (reference_has_nan(mapped)) { return vec3f(1.0, 0.0, 1.0); }
    let positive = reference_has_pos_inf(mapped);
    let negative = reference_has_neg_inf(mapped);
    if (positive || negative) {
        return reference_exception_color(positive, negative);
    }
    if (parameters.showAnomalies != 0u) {
        if (any(mapped < vec3f(0.0))) { return vec3f(0.0, 0.25, 1.0); }
        if (any(mapped > vec3f(parameters.linearOutputPeak))) { return vec3f(1.0, 0.05, 0.0); }
    }
    return reference_srgb_encode(clamp(mapped, vec3f(0.0), vec3f(parameters.linearOutputPeak)));
}
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let pixel = id.xy;
    if (pixel.x >= parameters.width || pixel.y >= parameters.height) { return; }
    let uv = (vec2f(pixel) + vec2f(0.5)) / vec2f(f32(parameters.width), f32(parameters.height));
    let ap0 = textureSampleLevel(inputTexture, inputSampler, uv, 0.0).rgb * parameters.exposureMultiplier;
    let mapped = reference_transform(ap0, parameters.linearOutputPeak);
    textureStore(outputTexture, vec2i(pixel), vec4f(reference_prepare_output(ap0, mapped), 1.0));
}
