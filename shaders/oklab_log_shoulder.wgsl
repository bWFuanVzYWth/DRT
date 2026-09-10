// SPDX-License-Identifier: GPL-3.0-only
// Oklab Log Shoulder DRT developed by this project's authors.
// Maps L^3 with a linear segment and log1p sigmoid shoulder, then softly compresses
// chroma against the gamut boundary along a fixed Oklab hue direction.
// Input: scene-linear ACES2065-1 (AP0). Output: display-encoded sRGB.

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
    rgbLogSigmoidBlackHueRetention: f32,
    rgbLogSigmoidWhiteHueRetention: f32,
    linearCompressionStart: f32,
    logSigmoidMaximumLogCoordinate: f32,
    logSigmoidOutputPeak: f32,
    rgbGamutExpansion: f32,
    linearSlope: f32,
    linearOutputPeak: f32,
    rgbHueRetention: f32,
    linearCurvePeak: f32,
}

@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var inputSampler: sampler;
@group(0) @binding(2) var outputTexture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> parameters: DrtParameters;

const OKLAB_LOG_SHOULDER_RGB_HEADROOM: f32 = 0.99999;
const OKLAB_LOG_SHOULDER_RED_ROW: vec3f = vec3f(4.0767416621, -3.3077115913, 0.2309699292);
const OKLAB_LOG_SHOULDER_GREEN_ROW: vec3f = vec3f(-1.2684380046, 2.6097574011, -0.3413193965);
const OKLAB_LOG_SHOULDER_BLUE_ROW: vec3f = vec3f(-0.0041960863, -0.7034186147, 1.7076147010);

fn encodeSrgb(linearRgb: vec3f) -> vec3f {
    let cutoff: vec3<bool> = linearRgb < vec3f(0.0031308);
    let higher: vec3f = 1.055 * pow(max(linearRgb, vec3f(0.0)), vec3f(1.0 / 2.4)) - vec3f(0.055);
    let lower: vec3f = linearRgb * 12.92;
    return vec3f(
        select(higher.x, lower.x, cutoff.x),
        select(higher.y, lower.y, cutoff.y),
        select(higher.z, lower.z, cutoff.z));
}

fn rgbToOklab(color: vec3f) -> vec3f {
    let l: f32 = 0.4122214708 * color.r + 0.5363325363 * color.g + 0.0514459929 * color.b;
    let m: f32 = 0.2119034982 * color.r + 0.6806995451 * color.g + 0.1073969566 * color.b;
    let s: f32 = 0.0883024619 * color.r + 0.2817188376 * color.g + 0.6299787005 * color.b;
    let lms: vec3f = pow(vec3f(l, m, s), vec3f(1.0 / 3.0));
    return vec3f(
        0.2104542553 * lms.x + 0.7936177850 * lms.y - 0.0040720468 * lms.z,
        1.9779984951 * lms.x - 2.4285922050 * lms.y + 0.4505937099 * lms.z,
        0.0259040371 * lms.x + 0.7827717662 * lms.y - 0.8086757660 * lms.z);
}

fn oklabToRgb(color: vec3f) -> vec3f {
    let lRoot: f32 = color.x + 0.3963377774 * color.y + 0.2158037573 * color.z;
    let mRoot: f32 = color.x - 0.1055613458 * color.y - 0.0638541728 * color.z;
    let sRoot: f32 = color.x - 0.0894841775 * color.y - 1.2914855480 * color.z;
    let lms: vec3f = vec3f(lRoot * lRoot * lRoot, mRoot * mRoot * mRoot, sRoot * sRoot * sRoot);
    return vec3f(
        dot(OKLAB_LOG_SHOULDER_RED_ROW, lms),
        dot(OKLAB_LOG_SHOULDER_GREEN_ROW, lms),
        dot(OKLAB_LOG_SHOULDER_BLUE_ROW, lms));
}

fn logDistance(distance: f32) -> f32 {
    // ln(1+x) = ln(2)*log2(1+x). Avoid cancellation close to the join.
    if (distance < 0.001) {
        return distance * (1.0 + distance * (-0.5 + distance / 3.0));
    }
    return 0.69314718056 * log2(1.0 + distance);
}

fn logShoulderComponent(value: f32) -> f32 {
    let slope = parameters.linearSlope;
    let start = parameters.linearCompressionStart;
    if (value <= start) {
        return slope * value;
    }
    let join = slope * start;
    let extent = parameters.linearOutputPeak - join;
    // Normalize the remaining output room before entering log space. This
    // supports a zero start while retaining the incoming linear slope.
    let distance = logDistance(slope * (value - start) / extent);
    let power = parameters.sigmoidShoulderPower;
    return join + extent * distance
        * pow(1.0 + parameters.sigmoidShoulderCoefficient * pow(distance, power), -1.0 / power);
}

fn mapLightness(lightness: f32) -> f32 {
    // Apply the tangent-continuous linear/log shoulder in L^3, then return
    // to Oklab lightness for the same fixed-hue chroma mapping as Oklab Reinhard.
    let brightness: f32 = lightness * lightness * lightness;
    return pow(logShoulderComponent(brightness), 1.0 / 3.0);
}

fn rootDirection(hue: vec2f) -> vec3f {
    return vec3f(
        0.3963377774 * hue.x + 0.2158037573 * hue.y,
        -0.1055613458 * hue.x - 0.0638541728 * hue.y,
        -0.0894841775 * hue.x - 1.2914855480 * hue.y);
}

fn maxSaturation(hue: vec2f, direction: vec3f) -> f32 {
    var k0: f32;
    var k1: f32;
    var k2: f32;
    var k3: f32;
    var k4: f32;
    var rgbRow: vec3f;
    if (-1.88170328 * hue.x - 0.80936493 * hue.y > 1.0) {
        k0 = 1.19086277; k1 = 1.76576728; k2 = 0.59662641; k3 = 0.75515197; k4 = 0.56771245;
        rgbRow = OKLAB_LOG_SHOULDER_RED_ROW;
    } else if (1.81444104 * hue.x - 1.19445276 * hue.y > 1.0) {
        k0 = 0.73956515; k1 = -0.45954404; k2 = 0.08285427; k3 = 0.12541070; k4 = 0.14503204;
        rgbRow = OKLAB_LOG_SHOULDER_GREEN_ROW;
    } else {
        k0 = 1.35733652; k1 = -0.00915799; k2 = -1.15130210; k3 = -0.50559606; k4 = 0.00692167;
        rgbRow = OKLAB_LOG_SHOULDER_BLUE_ROW;
    }

    let saturation: f32 = k0 + k1 * hue.x + k2 * hue.y + k3 * hue.x * hue.x + k4 * hue.x * hue.y;
    let roots: vec3f = vec3f(1.0) + saturation * direction;
    let lms: vec3f = roots * roots * roots;
    let firstLms: vec3f = 3.0 * direction * roots * roots;
    let secondLms: vec3f = 6.0 * direction * direction * roots;
    let f: f32 = dot(rgbRow, lms);
    let f1: f32 = dot(rgbRow, firstLms);
    let f2: f32 = dot(rgbRow, secondLms);
    return saturation - f * f1 / (f1 * f1 - 0.5 * f * f2);
}

fn connectedSaturation(hue: vec2f, saturation: f32) -> f32 {
    const blueNotchAxis: vec2f = vec2f(-0.10362546, -0.99461639);
    let alignment: f32 = max(dot(hue, blueNotchAxis), 0.0);
    let alignment2: f32 = alignment * alignment;
    let alignment4: f32 = alignment2 * alignment2;
    let alignment8: f32 = alignment4 * alignment4;
    let alignment16: f32 = alignment8 * alignment8;
    let alignment32: f32 = alignment16 * alignment16;
    let alignment64: f32 = alignment32 * alignment32;
    let alignment128: f32 = alignment64 * alignment64;
    let alignment256: f32 = alignment128 * alignment128;
    return min(saturation, 0.57 + (1.0 - alignment256));
}

fn cuspLightness(saturation: f32, direction: vec3f) -> f32 {
    let roots: vec3f = vec3f(1.0) + saturation * direction;
    let lms: vec3f = roots * roots * roots;
    let rgb: vec3f = vec3f(
        dot(OKLAB_LOG_SHOULDER_RED_ROW, lms),
        dot(OKLAB_LOG_SHOULDER_GREEN_ROW, lms),
        dot(OKLAB_LOG_SHOULDER_BLUE_ROW, lms));
    return pow(1.0 / max(rgb.r, max(rgb.g, rgb.b)), 1.0 / 3.0);
}

fn refineUpperChroma(chroma: f32, lightness: f32, direction: vec3f) -> f32 {
    let roots: vec3f = vec3f(lightness) + chroma * direction;
    let lms: vec3f = roots * roots * roots;
    let firstLms: vec3f = 3.0 * direction * roots * roots;
    let secondLms: vec3f = 6.0 * direction * direction * roots;
    let rgb: vec3f = vec3f(dot(OKLAB_LOG_SHOULDER_RED_ROW, lms), dot(OKLAB_LOG_SHOULDER_GREEN_ROW, lms), dot(OKLAB_LOG_SHOULDER_BLUE_ROW, lms));
    let firstRgb: vec3f = vec3f(dot(OKLAB_LOG_SHOULDER_RED_ROW, firstLms), dot(OKLAB_LOG_SHOULDER_GREEN_ROW, firstLms), dot(OKLAB_LOG_SHOULDER_BLUE_ROW, firstLms));
    let secondRgb: vec3f = vec3f(dot(OKLAB_LOG_SHOULDER_RED_ROW, secondLms), dot(OKLAB_LOG_SHOULDER_GREEN_ROW, secondLms), dot(OKLAB_LOG_SHOULDER_BLUE_ROW, secondLms));
    let f: vec3f = rgb - vec3f(1.0);
    let denominator: vec3f = firstRgb * firstRgb - 0.5 * f * secondRgb;
    let reciprocalStep: vec3f = firstRgb / denominator;
    var step: vec3f = -f * reciprocalStep;
    step = vec3f(
        select(1.0e20, step.x, reciprocalStep.x >= 0.0),
        select(1.0e20, step.y, reciprocalStep.y >= 0.0),
        select(1.0e20, step.z, reciprocalStep.z >= 0.0));
    return chroma + min(step.r, min(step.g, step.b));
}

fn softMin(value: f32, limit: f32, power: f32) -> f32 {
    if (value <= 0.0 || limit <= 0.0) {
        return 0.0;
    }
    let lower: f32 = min(value, limit);
    let higher: f32 = max(value, limit);
    let ratio: f32 = lower / higher;
    return lower * pow(1.0 + pow(ratio, power), -1.0 / power);
}

fn softMin4(value: f32, limit: f32) -> f32 {
    if (value <= 0.0 || limit <= 0.0) {
        return 0.0;
    }
    let lower: f32 = min(value, limit);
    let higher: f32 = max(value, limit);
    let ratio: f32 = lower / higher;
    let ratio2: f32 = ratio * ratio;
    let root: f32 = sqrt(1.0 + ratio2 * ratio2);
    return lower * inverseSqrt(root);
}

fn saturationCap(lightness: f32, maximumSaturation: f32, direction: vec3f) -> f32 {
    if (lightness <= 0.0) {
        return maximumSaturation;
    }
    if (lightness >= 1.0) {
        return 0.0;
    }
    let cusp: f32 = cuspLightness(maximumSaturation, direction);
    let blackChroma: f32 = lightness * maximumSaturation;
    var whiteChroma: f32 = cusp * maximumSaturation * (1.0 - lightness) / (1.0 - cusp);
    whiteChroma = refineUpperChroma(whiteChroma, lightness, direction);
    let t: f32 = clamp((lightness - cusp) / (1.0 - cusp), 0.0, 1.0);
    let shoulder: f32 = t * (1.0 - t);
    whiteChroma *= 1.0 - 0.0035 * 16.0 * shoulder * shoulder;
    let roundedChroma: f32 = softMin4(blackChroma, whiteChroma);
    return max(roundedChroma / lightness, 0.0);
}

fn chromaRetention(lightness: f32) -> f32 {
    let lightness2: f32 = lightness * lightness;
    let lightness4: f32 = lightness2 * lightness2;
    let lightness8: f32 = lightness4 * lightness4;
    return 1.0 - lightness8 * lightness4;
}

fn roundingPower(lightness: f32) -> f32 {
    let endpointDistance: f32 = lightness * (1.0 - lightness);
    return 32.0 - 256.0 * endpointDistance * endpointDistance;
}

fn mapLinearRgb(color: vec3f) -> vec3f {
    let oklab: vec3f = rgbToOklab(color);
    if (oklab.x <= 0.0) {
        return vec3f(0.0);
    }
    let outputLightness: f32 = mapLightness(oklab.x);
    let inputChroma: f32 = length(oklab.yz);
    if (inputChroma <= 1.0e-8) {
        return OKLAB_LOG_SHOULDER_RGB_HEADROOM * oklabToRgb(vec3f(outputLightness, 0.0, 0.0));
    }
    let hue: vec2f = oklab.yz / inputChroma;
    let direction: vec3f = rootDirection(hue);
    let inputSaturation: f32 = inputChroma / oklab.x;
    let maximumSaturation: f32 = connectedSaturation(hue, maxSaturation(hue, direction));
    let desiredSaturation: f32 = inputSaturation * chromaRetention(outputLightness);
    let cap: f32 = saturationCap(outputLightness, maximumSaturation, direction);
    let outputSaturation: f32 = softMin(desiredSaturation, cap, roundingPower(outputLightness));
    return OKLAB_LOG_SHOULDER_RGB_HEADROOM * oklabToRgb(vec3f(
        outputLightness, outputLightness * outputSaturation * hue));
}

fn acesAp0ToRec709(ap0: vec3f) -> vec3f {
    return vec3f(
        dot(ap0, vec3f(2.5214008886, -1.1339957494, -0.3875618568)),
        dot(ap0, vec3f(-0.2762140616, 1.3725955663, -0.0962823557)),
        dot(ap0, vec3f(-0.0153202001, -0.1529925618, 1.1683871996)));
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
    // Magenta is reserved for undefined values in both modes.
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
    if (parameters.showAnomalies != 0) {
        // Distinct diagnostic classes for finite range errors.
        if (any(mapped < vec3f(0.0))) {
            return vec3f(0.0, 0.25, 1.0);
        }
        if (any(mapped > vec3f(1.0))) {
            return vec3f(1.0, 0.05, 0.0);
        }
    }

    return clamp(encodeSrgb(clamp(mapped, vec3f(0.0), vec3f(1.0))), vec3f(0.0), vec3f(1.0));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) dispatchThreadId: vec3u) {
    let pixel: vec2u = dispatchThreadId.xy;
    if (pixel.x >= parameters.width || pixel.y >= parameters.height) { return; }
    let uv: vec2f = (vec2f(pixel) + vec2f(0.5)) / vec2f(f32(parameters.width), f32(parameters.height));
    let ap0: vec3f = textureSampleLevel(inputTexture, inputSampler, uv, 0.0).rgb * parameters.exposureMultiplier;
    let mapped: vec3f = mapLinearRgb(acesAp0ToRec709(ap0));
    textureStore(outputTexture, vec2i(pixel), vec4f(prepareOutput(ap0, mapped), 1.0));
}
