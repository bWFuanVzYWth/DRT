// RGB Reinhard: virtual RGB coordinates, a per-channel linear segment joined
// to a Reinhard shoulder, and optional HSV hue repair after sRGB encoding.
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

fn toExpandedGamutCoordinates(color: vec3f) -> vec3f {
    // Moving the virtual primaries outward makes the coordinates of an
    // unchanged Rec.709 color contract toward the neutral axis.
    let neutral: f32 = dot(color, NEUTRAL_WEIGHTS);
    return mix(color, vec3f(neutral), parameters.rgbGamutExpansion);
}

fn fromExpandedGamutCoordinates(color: vec3f) -> vec3f {
    // The neutral projection is invariant, making this the exact inverse of
    // toExpandedGamutCoordinates before the intervening nonlinear curve.
    let neutral: f32 = dot(color, NEUTRAL_WEIGHTS);
    return (color - parameters.rgbGamutExpansion * vec3f(neutral))
        / (1.0 - parameters.rgbGamutExpansion);
}

fn reinhard(color: vec3f) -> vec3f {
    let linearSlope: f32 = parameters.linearSlope;
    let compressionStart: f32 = parameters.linearCompressionStart;
    let startOutput: f32 = linearSlope * compressionStart;
    let shoulderExtent: f32 = parameters.linearCurvePeak - startOutput;
    // The shoulder is unused below the join; keep its denominator positive there.
    let distance: vec3f = max(color - vec3f(compressionStart), vec3f(0.0));
    let tangentDistance: vec3f = linearSlope * distance;
    let linear: vec3f = linearSlope * color;
    let shoulder: vec3f = vec3f(startOutput)
        + tangentDistance / (vec3f(1.0) + tangentDistance / shoulderExtent);
    return vec3f(
        select(shoulder.x, linear.x, color.x <= compressionStart),
        select(shoulder.y, linear.y, color.y <= compressionStart),
        select(shoulder.z, linear.z, color.z <= compressionStart));
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

fn protectHsvHue(originalLinear: vec3f, mappedDisplay: vec3f) -> vec3f {
    if (parameters.rgbHueRetention <= 0.0) {
        return mappedDisplay;
    }
    let originalHsv: vec3f = rgbToHsv(encodeSrgb(originalLinear));
    var mappedHsv: vec3f = rgbToHsv(mappedDisplay);
    if (originalHsv.y <= 1.0e-7 || mappedHsv.y <= 1.0e-7) {
        return mappedDisplay;
    }
    var hueOffset: f32 = originalHsv.x - mappedHsv.x;
    hueOffset -= floor(hueOffset + 0.5);
    mappedHsv.x = fract(
        mappedHsv.x + parameters.rgbHueRetention * hueOffset);
    return hsvToRgb(mappedHsv);
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
    let ap0: vec3f = textureSampleLevel(inputTexture, inputSampler, uv, 0.0).rgb
        * parameters.exposureMultiplier;
    let linearRec709: vec3f = max(acesAp0ToRec709(ap0), vec3f(0.0));
    let working: vec3f = toExpandedGamutCoordinates(linearRec709);
    let mappedLinear: vec3f = fromExpandedGamutCoordinates(reinhard(working));
    var mappedDisplay: vec3f = encodeSrgb(mappedLinear);
    mappedDisplay = protectHsvHue(linearRec709, mappedDisplay);
    textureStore(outputTexture, vec2i(pixel), vec4f(prepareOutput(ap0, mappedDisplay), 1.0));
}
