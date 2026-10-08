// RGB Log Sigmoid: a per-channel log2 curve in virtual RGB coordinates.
// Below the adjustable join, the curve encodes an exact linear-light segment analytically;
// above it, a tangent-matched AgX-form sigmoid shoulder compresses highlights.
// Inset/outset and HSV hue repair can still change colored shadows.
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

const NEUTRAL_WEIGHTS: vec3f = vec3f(
    0.2120053547549465,
    0.3921825078090138,
    0.3958121374360396);

fn acesAp0ToRec709(ap0: vec3f) -> vec3f {
    return vec3f(
        dot(ap0, vec3f(2.5214008886, -1.1339957494, -0.3875618568)),
        dot(ap0, vec3f(-0.2762140616, 1.3725955663, -0.0962823557)),
        dot(ap0, vec3f(-0.0153202001, -0.1529925618, 1.1683871996)));
}

fn rgbToHsv(color: vec3f) -> vec3f {
    let maximum: f32 = max(color.r, max(color.g, color.b));
    let minimum: f32 = min(color.r, min(color.g, color.b));
    let chroma: f32 = maximum - minimum;
    var hue: f32 = 0.0;
    if (chroma > 1.0e-7) {
        if (maximum == color.r) {
            hue = (color.g - color.b) / chroma;
        } else if (maximum == color.g) {
            hue = (color.b - color.r) / chroma + 2.0;
        } else {
            hue = (color.r - color.g) / chroma + 4.0;
        }
        hue = fract(hue / 6.0);
    }
    let saturation: f32 = select(0.0, chroma / max(maximum, 1.0e-7), maximum > 1.0e-7);
    return vec3f(hue, saturation, maximum);
}

fn hsvToRgb(hsv: vec3f) -> vec3f {
    let primary: vec3f = clamp(
        abs(fract(vec3f(hsv.x) + vec3f(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - vec3f(3.0)) - vec3f(1.0),
        vec3f(0.0),
        vec3f(1.0));
    return hsv.z * mix(vec3f(1.0), primary, hsv.y);
}

fn toVirtualRgb(color: vec3f) -> vec3f {
    let neutral: f32 = dot(color, NEUTRAL_WEIGHTS);
    return mix(color, vec3f(neutral), parameters.logSigmoidGamutCompression);
}

fn fromVirtualRgb(color: vec3f) -> vec3f {
    let neutral: f32 = dot(color, NEUTRAL_WEIGHTS);
    return (color - parameters.logSigmoidGamutCompression * vec3f(neutral))
        / (1.0 - parameters.logSigmoidGamutCompression);
}

fn logSigmoidComponent(value: f32) -> f32 {
    let distance: f32 = value - parameters.logSigmoidInputPivot;
    if (distance <= 0.0) {
        // F(u) = OETF(k * join * 2^((u - pivot) * dynamicRange)).
        // Decoding F recovers k*x; the host derives the shoulder's tangent
        // from this same expression, so value and first derivative agree.
        let linearPivot: f32 = decodeSrgb(vec3f(parameters.logSigmoidOutputPivot)).x;
        let linearValue: f32 = linearPivot
            * exp2(distance / parameters.logSigmoidInverseDynamicRange);
        return encodeSrgb(vec3f(linearValue)).x;
    }
    let power: f32 = parameters.sigmoidShoulderPower;
    let coefficient: f32 = parameters.sigmoidShoulderCoefficient;
    return parameters.logSigmoidOutputPivot
        + parameters.logSigmoidPivotSlope * distance
            * pow(1.0 + coefficient * pow(distance, power), -1.0 / power);
}

fn linearToLogSigmoidComponent(value: f32) -> f32 {
    if (value <= 0.0) { return 0.0; }
    // No lower log clamp: the coordinate origin is not a black floor.
    let normalizedLog: f32 = min(
        (log2(value) - parameters.logSigmoidMinimumLog2)
            * parameters.logSigmoidInverseDynamicRange,
        parameters.logSigmoidMaximumLogCoordinate);
    return logSigmoidComponent(normalizedLog);
}

fn logSigmoidCurve(value: vec3f) -> vec3f {
    return vec3f(
        linearToLogSigmoidComponent(value.x),
        linearToLogSigmoidComponent(value.y),
        linearToLogSigmoidComponent(value.z));
}

fn rgbLogSigmoid(linearRec709: vec3f) -> vec3f {
    // TODO: Known issue: a strong per-channel shoulder can produce perceptual
    // banding across high-to-low-saturation highlight transitions. The current
    // 5.2 default is an accepted artistic compromise pending a color-trajectory
    // model that is independent from the tone curve.
    let inset: vec3f = toVirtualRgb(linearRec709);
    return fromVirtualRgb(logSigmoidCurve(inset));
}

fn adjustHsv(originalLinear: vec3f, mappedDisplay: vec3f) -> vec3f {
    // Compare both hues in the same pure-2.2 signal domain used by AgX Base.
    let originalDisplay: vec3f = pow(originalLinear, vec3f(1.0 / 2.2));
    let originalHsv: vec3f = rgbToHsv(originalDisplay);
    var mappedHsv: vec3f = rgbToHsv(mappedDisplay);

    if (originalHsv.y > 1.0e-7 && mappedHsv.y > 1.0e-7) {
        var hueOffset: f32 = originalHsv.x - mappedHsv.x;
        hueOffset -= floor(hueOffset + 0.5);
        mappedHsv.x = fract(mappedHsv.x + parameters.rgbLogSigmoidHueRetention * hueOffset);
    }

    mappedHsv.y = clamp(mappedHsv.y, 0.0, 1.0);
    return hsvToRgb(mappedHsv);
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

fn decodeSrgb(encoded: vec3f) -> vec3f {
    let lower: vec3f = encoded / 12.92;
    let higher: vec3f = pow(max((encoded + vec3f(0.055)) / 1.055, vec3f(0.0)), vec3f(2.4));
    return select(higher, lower, encoded <= vec3f(0.04045));
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
    if (parameters.showAnomalies != 0) {
        if (any(mapped < vec3f(0.0))) {
            return vec3f(0.0, 0.25, 1.0);
        }
        if (any(mapped > vec3f(parameters.logSigmoidOutputPeak))) {
            return vec3f(1.0, 0.05, 0.0);
        }
    }
    return clamp(mapped, vec3f(0.0), vec3f(parameters.logSigmoidOutputPeak));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) dispatchThreadId: vec3u) {
    let pixel: vec2u = dispatchThreadId.xy;
    if (pixel.x >= parameters.width || pixel.y >= parameters.height) { return; }
    let uv: vec2f = (vec2f(pixel) + vec2f(0.5)) / vec2f(f32(parameters.width), f32(parameters.height));
    let ap0: vec3f = textureSampleLevel(inputTexture, inputSampler, uv, 0.0).rgb * parameters.exposureMultiplier;
    let originalLinear: vec3f = max(acesAp0ToRec709(ap0), vec3f(0.0));
    let mapped: vec3f = adjustHsv(originalLinear, rgbLogSigmoid(originalLinear));
    textureStore(outputTexture, vec2i(pixel), vec4f(prepareOutput(ap0, mapped), 1.0));
}
