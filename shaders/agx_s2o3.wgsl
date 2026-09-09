// SPDX-License-Identifier: GPL-3.0-only
// AgX-S2O3 analytical display transform by linlin, ported from the author's implementation:
// C:/WorkSpace/AgX/agx.glsl at commit 0796e1b4aa9df94152eff353bae131eae1a4c087.
// Copyright (c) 2024 linlin.
// Input: scene-linear ACES2065-1 (AP0). Output: display-encoded BT.709/sRGB.

// Keep field order in sync with Parameters in src/gpu.rs.
struct DrtParameters {
    exposureMultiplier: f32,
    width: u32,
    height: u32,
    showAnomalies: u32,
    agxMinimumLog2: f32,
    agxInverseDynamicRange: f32,
    agxInputPivot: f32,
    agxOutputPivot: f32,
    agxPivotSlope: f32,
    agxToePower: f32,
    agxShoulderPower: f32,
    agxGamutCompression: f32,
    agxToeA: f32,
    agxShoulderA: f32,
    agxBlackHueRetention: f32,
    agxWhiteHueRetention: f32,
    reinhardCompressionStart: f32,
    agxMaximumLogCoordinate: f32,
    agxOutputPeak: f32,
    reinhardGamutExpansion: f32,
    reinhardLinearSlope: f32,
    reinhardOutputPeak: f32,
    reinhardHueRetention: f32,
    reinhardCurvePeak: f32,
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

const AGX_NEUTRAL_WEIGHTS: vec3f = vec3f(
    0.2120053547549465,
    0.3921825078090138,
    0.3958121374360396);

fn agxInset(color: vec3f) -> vec3f {
    let neutral: f32 = dot(color, AGX_NEUTRAL_WEIGHTS);
    return mix(color, vec3f(neutral), parameters.agxGamutCompression);
}

fn agxOutset(color: vec3f) -> vec3f {
    let neutral: f32 = dot(color, AGX_NEUTRAL_WEIGHTS);
    return (color - parameters.agxGamutCompression * vec3f(neutral))
        / (1.0 - parameters.agxGamutCompression);
}

fn agxCurveComponent(value: f32) -> f32 {
    let toe: bool = value <= parameters.agxInputPivot;
    let power: f32 = select(parameters.agxShoulderPower, parameters.agxToePower, toe);
    let coefficient: f32 = select(parameters.agxShoulderA, parameters.agxToeA, toe);
    let distance: f32 = value - parameters.agxInputPivot;
    return parameters.agxOutputPivot
        + parameters.agxPivotSlope * distance
            * pow(1.0 + coefficient * pow(abs(distance), power), -1.0 / power);
}

fn agxCurve(value: vec3f) -> vec3f {
    return vec3f(
        agxCurveComponent(value.x),
        agxCurveComponent(value.y),
        agxCurveComponent(value.z));
}

fn agxS2O3(linearRec709: vec3f) -> vec3f {
    // The original AgX-S2O3 OCIO chain applies RangeTransform(minIn=0,
    // minOut=0) before the inset. AP0-to-Rec.709 conversion can produce valid
    // negative wide-gamut components, so enforce that input-domain contract
    // here before taking log2.
    let inset: vec3f = agxInset(max(linearRec709, vec3f(0.0)));
    let normalizedLog: vec3f = clamp(
        (log2(inset) - vec3f(parameters.agxMinimumLog2))
            * parameters.agxInverseDynamicRange,
        vec3f(0.0),
        vec3f(1.0));
    return agxOutset(agxCurve(normalizedLog));
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
        if (any(mapped > vec3f(1.0))) {
            return vec3f(1.0, 0.05, 0.0);
        }
    }

    // agxS2O3 already returns display-encoded BT.709; do not apply another OETF.
    return clamp(mapped, vec3f(0.0), vec3f(1.0));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) dispatchThreadId: vec3u) {
    let pixel: vec2u = dispatchThreadId.xy;
    if (pixel.x >= parameters.width || pixel.y >= parameters.height) { return; }
    let uv: vec2f = (vec2f(pixel) + vec2f(0.5)) / vec2f(f32(parameters.width), f32(parameters.height));
    let ap0: vec3f = textureSampleLevel(inputTexture, inputSampler, uv, 0.0).rgb * parameters.exposureMultiplier;
    let mapped: vec3f = agxS2O3(acesAp0ToRec709(ap0));
    textureStore(outputTexture, vec2i(pixel), vec4f(prepareOutput(ap0, mapped), 1.0));
}
