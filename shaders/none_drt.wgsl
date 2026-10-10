// Display conversion baseline with no tone mapping.
// Input: scene-linear ACES2065-1 (AP0).
// Output: display-encoded extended sRGB (1.0 is SDR reference white).

// Keep field order in sync with Parameters in src/gpu.rs.
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
}

@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var inputSampler: sampler;
@group(0) @binding(2) var outputTexture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> parameters: DrtParameters;

fn acesAp0ToRec709(ap0: vec3f) -> vec3f {
    return vec3f(
        dot(ap0, vec3f(2.5214008886, -1.1339957494, -0.3875618568)),
        dot(ap0, vec3f(-0.2762140616, 1.3725955663, -0.0962823557)),
        dot(ap0, vec3f(-0.0153202001, -0.1529925618, 1.1683871996)));
}

fn encodeSrgb(linearRgb: vec3f) -> vec3f {
    let cutoff: vec3<bool> = linearRgb < vec3f(0.0031308);
    let higher: vec3f = 1.055 * pow(max(linearRgb, vec3f(0.0)), vec3f(1.0 / 2.4)) - vec3f(0.055);
    let lower: vec3f = linearRgb * 12.92;
    return vec3f(
        select(higher.x, lower.x, cutoff.x),
        select(higher.y, lower.y, cutoff.y),
        select(higher.z, lower.z, cutoff.z));
}

// Classify IEEE-754 bits; WGSL has no isnan/isinf built-ins.
fn hasNan(value: vec3f) -> bool {
    return any((bitcast<vec3u>(value) & vec3u(0x7fffffffu)) > vec3u(0x7f800000u));
}

fn hasPositiveInfinity(value: vec3f) -> bool {
    return any(bitcast<vec3u>(value) == vec3u(0x7f800000u));
}

fn hasNegativeInfinity(value: vec3f) -> bool {
    return any(bitcast<vec3u>(value) == vec3u(0xff800000u));
}

fn anomalyColor(positiveInfinity: bool, negativeInfinity: bool) -> vec3f {
    if (parameters.showAnomalies != 0) {
        if (positiveInfinity && negativeInfinity) {
            return vec3f(1.0, 0.35, 0.0);
        }
        if (positiveInfinity) {
            return vec3f(1.0, 1.0, 0.0);
        }
        return vec3f(0.0, 1.0, 1.0);
    }
    if (positiveInfinity && negativeInfinity) {
        return vec3f(1.0, 0.0, 1.0);
    }
    if (positiveInfinity) {
        return vec3f(1.0, 1.0, 1.0);
    }
    return vec3f(0.0, 0.0, 0.0);
}

fn prepareOutput(source: vec3f, mapped: vec3f) -> vec3f {
    if (hasNan(source)) {
        return vec3f(1.0, 0.0, 1.0);
    }
    var positiveInfinity: bool = hasPositiveInfinity(source);
    var negativeInfinity: bool = hasNegativeInfinity(source);
    if (positiveInfinity || negativeInfinity) {
        return anomalyColor(positiveInfinity, negativeInfinity);
    }
    if (hasNan(mapped)) {
        return vec3f(1.0, 0.0, 1.0);
    }
    positiveInfinity = hasPositiveInfinity(mapped);
    negativeInfinity = hasNegativeInfinity(mapped);
    if (positiveInfinity || negativeInfinity) {
        return anomalyColor(positiveInfinity, negativeInfinity);
    }
    let encodedOutputPeak: f32 = encodeSrgb(vec3f(parameters.linearOutputPeak)).x;
    if (parameters.showAnomalies != 0) {
        if (any(mapped < vec3f(0.0))) {
            return vec3f(0.0, 0.25, 1.0);
        }
        if (any(mapped > vec3f(encodedOutputPeak))) {
            return vec3f(1.0, 0.05, 0.0);
        }
    }
    return clamp(mapped, vec3f(0.0), vec3f(encodedOutputPeak));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) dispatchThreadId: vec3u) {
    let pixel: vec2u = dispatchThreadId.xy;
    if (pixel.x >= parameters.width || pixel.y >= parameters.height) { return; }
    let uv: vec2f = (vec2f(pixel) + vec2f(0.5)) / vec2f(f32(parameters.width), f32(parameters.height));
    let ap0: vec3f = textureSampleLevel(inputTexture, inputSampler, uv, 0.0).rgb * parameters.exposureMultiplier;
    let mapped: vec3f = encodeSrgb(acesAp0ToRec709(ap0));
    textureStore(outputTexture, vec2i(pixel), vec4f(prepareOutput(ap0, mapped), 1.0));
}
