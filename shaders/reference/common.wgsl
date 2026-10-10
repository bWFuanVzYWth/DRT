// SPDX-License-Identifier: GPL-3.0-only
// Workbench adapters shared by the reference ports. Algorithm provenance lives
// alongside each reference_transform fragment in references/entries/.
struct DrtParameters {
    exposureMultiplier: f32,
    width: u32,
    height: u32,
    showAnomalies: u32,
    logSigmoidMinimumLog2: f32,
    logSigmoidInverseDynamicRange: f32,
    logSigmoidInputPivot: f32,
    logSigmoidOutputPivot: f32,
    logSigmoidPivotSlope: f32,
    logSigmoidToePower: f32,
    sigmoidShoulderPower: f32,
    logSigmoidGamutCompression: f32,
    logSigmoidToeCoefficient: f32,
    sigmoidShoulderCoefficient: f32,
    rgbLogSigmoidHueRetention: f32,
    linearCompressionStart: f32,
    logSigmoidMaximumLogCoordinate: f32,
    logSigmoidOutputPeak: f32,
    rgbGamutExpansion: f32,
    linearSlope: f32,
    linearOutputPeak: f32,
    rgbHueRetention: f32,
    linearCurvePeak: f32,
    oklabHighlightChromaPower: f32,
    oklabGamutRoundingPower: f32,
    oklabEndpointCompressionPower: f32,
    oklabMidtoneCompressionPower: f32,
    oklabAcesLinearSlope: f32,
    oklabAcesCompressionStart: f32,
    oklabAcesShoulderPower: f32,
}
@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var inputSampler: sampler;
@group(0) @binding(2) var outputTexture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> parameters: DrtParameters;
@group(1) @binding(0) var<storage, read> reference_data: array<f32>;

fn reference_ap0_to_rec709(ap0: vec3f) -> vec3f {
    return vec3f(
        dot(ap0, vec3f(2.5214008886, -1.1339957494, -0.3875618568)),
        dot(ap0, vec3f(-0.2762140616, 1.3725955663, -0.0962823557)),
        dot(ap0, vec3f(-0.0153202001, -0.1529925618, 1.1683871996)));
}
fn reference_srgb_encode_channel(x: f32) -> f32 {
    let magnitude = abs(x);
    let encoded = select(1.055 * pow(magnitude, 1.0 / 2.4) - 0.055,
        magnitude * 12.92, magnitude <= 0.0031308);
    return sign(x) * encoded;
}
fn reference_srgb_encode(x: vec3f) -> vec3f {
    return vec3f(reference_srgb_encode_channel(x.r), reference_srgb_encode_channel(x.g), reference_srgb_encode_channel(x.b));
}
fn reference_srgb_decode_channel(x: f32) -> f32 {
    let magnitude = abs(x);
    return sign(x) * select(pow((magnitude + 0.055) / 1.055, 2.4), magnitude / 12.92, magnitude <= 0.04045);
}
fn reference_srgb_decode(x: vec3f) -> vec3f {
    return vec3f(reference_srgb_decode_channel(x.r), reference_srgb_decode_channel(x.g), reference_srgb_decode_channel(x.b));
}

fn reference_lut_fetch(offset: u32, size: u32, point: vec3u) -> vec3f {
    let index = offset + 3u * (point.x + size * (point.y + size * point.z));
    return vec3f(reference_data[index], reference_data[index + 1u], reference_data[index + 2u]);
}
// OCIO's tetrahedral interpolation, preserving the shipped LUT grid exactly.
fn reference_sample_cube(value: vec3f, size: u32, offset: u32) -> vec3f {
    let coordinate = clamp(value, vec3f(0.0), vec3f(1.0)) * f32(size - 1u);
    let low = vec3u(floor(coordinate));
    let high = min(low + vec3u(1u), vec3u(size - 1u));
    let f = coordinate - vec3f(low);
    let c000 = reference_lut_fetch(offset, size, low);
    let c111 = reference_lut_fetch(offset, size, high);
    if (f.x >= f.y) {
        let c100 = reference_lut_fetch(offset, size, vec3u(high.x, low.y, low.z));
        if (f.y >= f.z) {
            let c110 = reference_lut_fetch(offset, size, vec3u(high.x, high.y, low.z));
            return c000 + f.x * (c100 - c000) + f.y * (c110 - c100) + f.z * (c111 - c110);
        }
        let c101 = reference_lut_fetch(offset, size, vec3u(high.x, low.y, high.z));
        if (f.x >= f.z) {
            return c000 + f.x * (c100 - c000) + f.z * (c101 - c100) + f.y * (c111 - c101);
        }
        let c001 = reference_lut_fetch(offset, size, vec3u(low.x, low.y, high.z));
        return c000 + f.z * (c001 - c000) + f.x * (c101 - c001) + f.y * (c111 - c101);
    }
    let c010 = reference_lut_fetch(offset, size, vec3u(low.x, high.y, low.z));
    if (f.x >= f.z) {
        let c110 = reference_lut_fetch(offset, size, vec3u(high.x, high.y, low.z));
        return c000 + f.y * (c010 - c000) + f.x * (c110 - c010) + f.z * (c111 - c110);
    }
    let c011 = reference_lut_fetch(offset, size, vec3u(low.x, high.y, high.z));
    if (f.y >= f.z) {
        return c000 + f.y * (c010 - c000) + f.z * (c011 - c010) + f.x * (c111 - c011);
    }
    let c001 = reference_lut_fetch(offset, size, vec3u(low.x, low.y, high.z));
    return c000 + f.z * (c001 - c000) + f.y * (c011 - c001) + f.x * (c111 - c011);
}
fn reference_sample_curve(value: f32, offset: u32, size: u32) -> f32 {
    let coordinate = clamp(value, 0.0, 1.0) * f32(size - 1u);
    let low = u32(floor(coordinate));
    let high = min(low + 1u, size - 1u);
    return mix(reference_data[offset + low], reference_data[offset + high], coordinate - f32(low));
}
