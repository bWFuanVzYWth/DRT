// SPDX-License-Identifier: GPL-3.0-only
// Original Oklab experiment inspired by ACES 2's perceptual tone stage.
// Inspiration: https://docs.acescentral.com/system-components/output-transforms/
// Combines an extended highlight shoulder with fixed-hue chroma compression.
// No ACES transform port or white adaptation stage.
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

const OKLAB_ACES_RGB_HEADROOM: f32 = 0.99999;
const OKLAB_ACES_RED_ROW: vec3f = vec3f(4.0767416621, -3.3077115913, 0.2309699292);
const OKLAB_ACES_GREEN_ROW: vec3f = vec3f(-1.2684380046, 2.6097574011, -0.3413193965);
const OKLAB_ACES_BLUE_ROW: vec3f = vec3f(-0.0041960863, -0.7034186147, 1.7076147010);

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
        dot(lms, OKLAB_ACES_RED_ROW),
        dot(lms, OKLAB_ACES_GREEN_ROW),
        dot(lms, OKLAB_ACES_BLUE_ROW));
}

// Reuse the connected fixed-hue sRGB boundary from Oklab Reinhard. Normalize
// lightness by the HDR peak before evaluating this unit-display boundary.
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
        rgbRow = OKLAB_ACES_RED_ROW;
    } else if (1.81444104 * hue.x - 1.19445276 * hue.y > 1.0) {
        k0 = 0.73956515; k1 = -0.45954404; k2 = 0.08285427; k3 = 0.12541070; k4 = 0.14503204;
        rgbRow = OKLAB_ACES_GREEN_ROW;
    } else {
        k0 = 1.35733652; k1 = -0.00915799; k2 = -1.15130210; k3 = -0.50559606; k4 = 0.00692167;
        rgbRow = OKLAB_ACES_BLUE_ROW;
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
        dot(OKLAB_ACES_RED_ROW, lms),
        dot(OKLAB_ACES_GREEN_ROW, lms),
        dot(OKLAB_ACES_BLUE_ROW, lms));
    return pow(1.0 / max(rgb.r, max(rgb.g, rgb.b)), 1.0 / 3.0);
}

fn refineUpperChroma(chroma: f32, lightness: f32, direction: vec3f) -> f32 {
    let roots: vec3f = vec3f(lightness) + chroma * direction;
    let lms: vec3f = roots * roots * roots;
    let firstLms: vec3f = 3.0 * direction * roots * roots;
    let secondLms: vec3f = 6.0 * direction * direction * roots;
    let rgb: vec3f = vec3f(dot(OKLAB_ACES_RED_ROW, lms), dot(OKLAB_ACES_GREEN_ROW, lms), dot(OKLAB_ACES_BLUE_ROW, lms));
    let firstRgb: vec3f = vec3f(dot(OKLAB_ACES_RED_ROW, firstLms), dot(OKLAB_ACES_GREEN_ROW, firstLms), dot(OKLAB_ACES_BLUE_ROW, firstLms));
    let secondRgb: vec3f = vec3f(dot(OKLAB_ACES_RED_ROW, secondLms), dot(OKLAB_ACES_GREEN_ROW, secondLms), dot(OKLAB_ACES_BLUE_ROW, secondLms));
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
    let roundedChroma: f32 = softMin(blackChroma, whiteChroma, 4.0);
    return max(roundedChroma / lightness, 0.0);
}

fn roundingPower(lightness: f32) -> f32 {
    let endpointDistance: f32 = lightness * (1.0 - lightness);
    let midtoneWeight: f32 = clamp(16.0 * endpointDistance * endpointDistance, 0.0, 1.0);
    return mix(32.0, 16.0, midtoneWeight);
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
    if lab.x <= 0.0 {
        // Keep signed/invalid input available to the range diagnostics.
        return gain * color;
    }

    var outputBrightness = gain * brightness;
    if brightness > join {
        let outputJoin = gain * join;
        let extent = parameters.linearOutputPeak - outputJoin;
        let q = gain * (brightness - join) / extent;
        let progress = shoulderProgress(q);
        outputBrightness = outputJoin + extent * progress;
    }
    let outputLightness = pow(outputBrightness, 1.0 / 3.0);
    let inputChroma = length(lab.yz);
    if inputChroma <= 1.0e-8 {
        // The neutral axis follows the scalar shoulder without RGB margin or
        // an inverse-Oklab round trip changing the displayed tone curve.
        return vec3f(outputBrightness);
    }

    let hue = lab.yz / inputChroma;
    let direction = rootDirection(hue);
    let inputSaturation = inputChroma / lab.x;
    let sourceReference = maxSaturation(hue, direction);
    let maximumSaturation = connectedSaturation(hue, sourceReference);
    let normalizedLightness = clamp(
        outputLightness / pow(parameters.linearOutputPeak, 1.0 / 3.0), 0.0, 1.0);
    let cap = saturationCap(normalizedLightness, maximumSaturation, direction);
    let legacySaturation = softMin(inputSaturation, cap, roundingPower(normalizedLightness));

    // C/L stays unchanged under exposure. Preserve that source saturation rank
    // near white instead of collapsing every saturated color of one hue onto
    // the same tiny display cap. The source reference keeps its full boundary;
    // only the target cap uses the connected blue-notch boundary guard.
    let saturationAnchor = softMin(inputSaturation, sourceReference, 16.0)
        / max(sourceReference, 1.0e-8);
    let rankedSaturation = cap * saturationAnchor;
    let outputSaturation = mix(legacySaturation, rankedSaturation,
        smoothstep(0.90, 0.97, normalizedLightness));

    // Apply one continuous fixed-hue constraint from black to white. A hard
    // shadow bypass would jump at the shoulder join for saturated colors.
    // The shrinking display boundary provides the highlight chroma fade;
    // no separate shoulder-progress desaturation is needed.
    return OKLAB_ACES_RGB_HEADROOM * oklabToRgb(vec3f(
        outputLightness, outputLightness * outputSaturation * hue));
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
