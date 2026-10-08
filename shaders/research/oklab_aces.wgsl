// SPDX-License-Identifier: GPL-3.0-only
// Original Oklab experiment inspired by ACES 2's perceptual tone stage.
// Inspiration: https://docs.acescentral.com/system-components/output-transforms/
// No ACES transform port, white adaptation stage, or gamut-boundary compression.
// Input: scene-linear ACES2065-1/AP0. Output: extended sRGB display signal.
// Scalar parameter definitions and the matching curve are in src/oklab_aces.rs.

// Keep this prefix and appended experimental fields in sync with src/gpu.rs.
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

fn signedCubeRoot(value: vec3f) -> vec3f {
    // Wide-gamut input may have negative Rec.709 components or cone responses.
    // Their signed cube roots retain finite values without clipping the input.
    return sign(value) * pow(abs(value), vec3f(1.0 / 3.0));
}

fn rgbToOklab(color: vec3f) -> vec3f {
    let lms = signedCubeRoot(vec3f(
        dot(color, vec3f(0.4122214708, 0.5363325363, 0.0514459929)),
        dot(color, vec3f(0.2119034982, 0.6806995451, 0.1073969566)),
        dot(color, vec3f(0.0883024619, 0.2817188376, 0.6299787005))));
    return vec3f(
        dot(lms, vec3f(0.2104542553, 0.7936177850, -0.0040720468)),
        dot(lms, vec3f(1.9779984951, -2.4285922050, 0.4505937099)),
        dot(lms, vec3f(0.0259040371, 0.7827717662, -0.8086757660)));
}

fn oklabToRgb(color: vec3f) -> vec3f {
    let roots = vec3f(
        color.x + 0.3963377774 * color.y + 0.2158037573 * color.z,
        color.x - 0.1055613458 * color.y - 0.0638541728 * color.z,
        color.x - 0.0894841775 * color.y - 1.2914855480 * color.z);
    let lms = roots * roots * roots;
    return vec3f(
        dot(lms, vec3f(4.0767416621, -3.3077115913, 0.2309699292)),
        dot(lms, vec3f(-1.2684380046, 2.6097574011, -0.3413193965)),
        dot(lms, vec3f(-0.0041960863, -0.7034186147, 1.7076147010)));
}

fn shoulderProgress(q: f32) -> f32 {
    let power = parameters.oklabAcesShoulderPower;
    if q < 0.001 {
        // Stable near the linear join, without cancellation in 1-pow(...).
        return q * (1.0 - (power + 1.0) / (2.0 * power) * q
            + (power + 1.0) * (power + 2.0) / (6.0 * power * power) * q * q);
    }
    return 1.0 - pow(1.0 + q / power, -power);
}

fn mapLinearRgb(color: vec3f) -> vec3f {
    let lab = rgbToOklab(color);
    let brightness = lab.x * lab.x * lab.x;
    let gain = parameters.oklabAcesLinearSlope;
    let join = parameters.oklabAcesCompressionStart;
    if brightness <= join {
        // Keep every colored shadow exactly linear, including signed input.
        // This bypass also avoids introducing Oklab round-trip error here.
        return gain * color;
    }

    let outputJoin = gain * join;
    let extent = parameters.linearOutputPeak - outputJoin;
    let q = gain * (brightness - join) / extent;
    let progress = shoulderProgress(q);
    let outputBrightness = outputJoin + extent * progress;
    let outputLightness = pow(outputBrightness, 1.0 / 3.0);

    // Fixed baseline from the former controls: desaturation=1, protection=0.
    // Preserve the Oklab hue and fade chroma smoothly through the shoulder.
    let retention = pow(max(0.0, 1.0 - progress * progress), 1.0);
    let chromaScale = outputLightness / lab.x * retention;
    return oklabToRgb(vec3f(outputLightness, lab.yz * chromaScale));
}

fn acesAp0ToRec709(ap0: vec3f) -> vec3f {
    // The workbench's existing input-space conversion, shared with other DRTs.
    return vec3f(
        dot(ap0, vec3f(2.5214008886, -1.1339957494, -0.3875618568)),
        dot(ap0, vec3f(-0.2762140616, 1.3725955663, -0.0962823557)),
        dot(ap0, vec3f(-0.0153202001, -0.1529925618, 1.1683871996)));
}

fn encodeSrgb(linearRgb: vec3f) -> vec3f {
    let lower = 12.92 * linearRgb;
    let higher = 1.055 * pow(max(linearRgb, vec3f(0.0)), vec3f(1.0 / 2.4)) - vec3f(0.055);
    return select(higher, lower, linearRgb <= vec3f(0.0031308));
}

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
    if parameters.showAnomalies != 0u {
        if positiveInfinity && negativeInfinity { return vec3f(1.0, 0.35, 0.0); }
        if positiveInfinity { return vec3f(1.0, 1.0, 0.0); }
        return vec3f(0.0, 1.0, 1.0);
    }
    if positiveInfinity && negativeInfinity { return vec3f(1.0, 0.0, 1.0); }
    if positiveInfinity { return vec3f(1.0); }
    return vec3f(0.0);
}

fn prepareOutput(source: vec3f, mapped: vec3f) -> vec3f {
    // Source errors take priority even if model evaluation generated a NaN.
    if hasNan(source) { return vec3f(1.0, 0.0, 1.0); }
    var positiveInfinity = hasPositiveInfinity(source);
    var negativeInfinity = hasNegativeInfinity(source);
    if positiveInfinity || negativeInfinity {
        return anomalyColor(positiveInfinity, negativeInfinity);
    }
    if hasNan(mapped) { return vec3f(1.0, 0.0, 1.0); }
    positiveInfinity = hasPositiveInfinity(mapped);
    negativeInfinity = hasNegativeInfinity(mapped);
    if positiveInfinity || negativeInfinity {
        return anomalyColor(positiveInfinity, negativeInfinity);
    }
    let peak = parameters.linearOutputPeak;
    if parameters.showAnomalies != 0u {
        if any(mapped < vec3f(0.0)) { return vec3f(0.0, 0.25, 1.0); }
        if any(mapped > vec3f(peak)) { return vec3f(1.0, 0.05, 0.0); }
    }
    // Presentation clipping is deliberately separate from the experimental
    // transform; diagnostics above expose out-of-range colors before clipping.
    let encodedPeak = encodeSrgb(vec3f(peak));
    return clamp(encodeSrgb(clamp(mapped, vec3f(0.0), vec3f(peak))), vec3f(0.0), encodedPeak);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) dispatchThreadId: vec3u) {
    let pixel = dispatchThreadId.xy;
    if pixel.x >= parameters.width || pixel.y >= parameters.height { return; }
    let uv = (vec2f(pixel) + vec2f(0.5)) / vec2f(f32(parameters.width), f32(parameters.height));
    let ap0 = textureSampleLevel(inputTexture, inputSampler, uv, 0.0).rgb * parameters.exposureMultiplier;
    let mapped = mapLinearRgb(acesAp0ToRec709(ap0));
    textureStore(outputTexture, vec2i(pixel), vec4f(prepareOutput(ap0, mapped), 1.0));
}
