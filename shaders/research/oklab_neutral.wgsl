// SPDX-License-Identifier: GPL-3.0-only
// Original Oklab Neutral experiment: max-RGB shoulder and perceptual whitening.
// Inspiration: Khronos PBR Neutral's peak scaling and RGB-to-white structure:
// https://github.com/KhronosGroup/ToneMapping/blob/180b1a7bddec33f73fe41712a2963cc3ad8e5547/PBR_Neutral/pbrNeutral.glsl
// Oklab's cube-root LMS space and conversion matrices:
// https://bottosson.github.io/posts/oklab/
// This is an original combination, not an official Khronos or ACES port.
// Input adapter: scene-linear ACES2065-1/AP0 -> Rec.709, then clip negative RGB.
// The core operates only on nonnegative linear Rec.709; output is extended sRGB.
// A small common-mode RGB lift handles the nonconvex pure-blue boundary without
// cusp searches, gamut-boundary intersection, or iterative gamut compression.
// Scalar parameter definitions and the matching curve: src/oklab_neutral.rs.

// Preserve the shared 120-byte prefix and append this experiment's three fields.
// Keep the complete field order in sync with Parameters in src/gpu.rs.
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
    oklabNeutralLinearSlope: f32,
    oklabNeutralCompressionStart: f32,
    oklabNeutralShoulderPower: f32,
}

@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var inputSampler: sampler;
@group(0) @binding(2) var outputTexture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> parameters: DrtParameters;

fn rgbToLms(color: vec3f) -> vec3f {
    return vec3f(
        dot(color, vec3f(0.4122214708, 0.5363325363, 0.0514459929)),
        dot(color, vec3f(0.2119034982, 0.6806995451, 0.1073969566)),
        dot(color, vec3f(0.0883024619, 0.2817188376, 0.6299787005)));
}

fn lmsToRgb(color: vec3f) -> vec3f {
    return vec3f(
        dot(color, vec3f(4.0767416621, -3.3077115913, 0.2309699292)),
        dot(color, vec3f(-1.2684380046, 2.6097574011, -0.3413193965)),
        dot(color, vec3f(-0.0041960863, -0.7034186147, 1.7076147010)));
}

fn mapLinearRgb(color: vec3f) -> vec3f {
    let maximum = max(color.r, max(color.g, color.b));
    let gain = parameters.oklabNeutralLinearSlope;
    let join = parameters.oklabNeutralCompressionStart;
    if maximum <= join {
        // Every nonnegative colored shadow stays exactly exposure-linear.
        return gain * color;
    }

    let outputJoin = gain * join;
    let extent = parameters.linearOutputPeak - outputJoin;
    let q = gain * (maximum - join) / extent;
    let power = parameters.oklabNeutralShoulderPower;
    var tail: f32;
    var mappedPeak: f32;
    if q < 0.001 {
        // Match the CPU curve while avoiding cancellation at the linear join.
        let progress = q * (1.0 - (power + 1.0) / (2.0 * power) * q
            + (power + 1.0) * (power + 2.0) / (6.0 * power * power) * q * q);
        tail = 1.0 - progress;
        mappedPeak = outputJoin + extent * progress;
    } else {
        tail = pow(1.0 + q / power, -power);
        mappedPeak = parameters.linearOutputPeak - extent * tail;
    }

    if color.r == color.g && color.g == color.b {
        // Follow the scalar tone curve exactly, without matrix round-trip error.
        return vec3f(mappedPeak);
    }

    // One shoulder progress drives both the peak and the approach to white.
    // This quartic equals 1-w, where w=P^3*(3-2P), P=1-tail.
    // Its first two derivatives vanish at the linear join, while its white-end
    // derivative stays nonzero so chroma and peak tails decay at the same rate.
    // Evaluate retention directly instead of subtracting two near-one values.
    let retention = tail * (1.0 + tail * (3.0 + tail * (-5.0 + 2.0 * tail)));
    let normalized = color / maximum;
    let roots = pow(rgbToLms(normalized), vec3f(1.0 / 3.0));
    let whiteRoots = vec3f(1.0) + retention * (roots - vec3f(1.0));
    let raw = lmsToRgb(whiteRoots * whiteRoots * whiteRoots);

    // LMS-root interpolation preserves Oklab hue before this correction.
    // Its pure-blue segment can dip slightly below zero: max normalization
    // alone cannot repair that. A common-mode lift guarantees [0,1] RGB;
    // the final positive scaling guarantees the requested HDR peak.
    let delta = max(0.0, -min(raw.r, min(raw.g, raw.b)));
    let rawMaximum = max(raw.r, max(raw.g, raw.b));
    let unitColor = (raw + vec3f(delta)) / (rawMaximum + delta);
    return mappedPeak * unitColor;
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
    // Source errors take priority even when the input adapter or transform
    // changes their IEEE-754 classification, including infinity producing NaN.
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
    let encodedPeak = encodeSrgb(vec3f(peak));
    return clamp(encodeSrgb(clamp(mapped, vec3f(0.0), vec3f(peak))), vec3f(0.0), encodedPeak);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) dispatchThreadId: vec3u) {
    let pixel = dispatchThreadId.xy;
    if pixel.x >= parameters.width || pixel.y >= parameters.height { return; }
    let uv = (vec2f(pixel) + vec2f(0.5)) / vec2f(f32(parameters.width), f32(parameters.height));
    let ap0 = textureSampleLevel(inputTexture, inputSampler, uv, 0.0).rgb * parameters.exposureMultiplier;
    let rec709 = max(acesAp0ToRec709(ap0), vec3f(0.0));
    let mapped = mapLinearRgb(rec709);
    textureStore(outputTexture, vec2i(pixel), vec4f(prepareOutput(ap0, mapped), 1.0));
}
