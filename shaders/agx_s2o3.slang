// AgX-S2O3 analytical display transform, ported from:
// C:/WorkSpace/AgX/agx.glsl at commit 0796e1b4aa9df94152eff353bae131eae1a4c087.
// Copyright (c) 2024 linlin. MIT licensed; see THIRD_PARTY_LICENSES/AgX-S2O3.txt.
// Input: scene-linear ACES2065-1 (AP0). Output: display-encoded BT.709/sRGB.

struct DrtParameters
{
    float exposureMultiplier;
    float overexposure;
    uint width;
    uint height;
    uint showAnomalies;
    uint padding0;
    uint padding1;
    uint padding2;
    float agxMinimumLog2;
    float agxInverseDynamicRange;
    float agxInputPivot;
    float agxOutputPivot;
    float agxPivotSlope;
    float agxToePower;
    float agxShoulderPower;
    float agxGamutCompression;
    float agxToeA;
    float agxShoulderA;
    float agxBlackHueRetention;
    float agxWhiteHueRetention;
    float agxBlackGamutOnset;
    float oklabWhiteHueCompression;
    float oklabHueCompressionOnset;
    uint padding3;
};

[[vk::binding(0, 0)]] Texture2D<float4> inputTexture;
[[vk::binding(1, 0)]] SamplerState inputSampler;
[[vk::binding(2, 0)]] [vk::image_format("rgba8")] WTexture2D<float4> outputTexture;
[[vk::binding(3, 0)]] ConstantBuffer<DrtParameters> parameters;

float3 acesAp0ToRec709(float3 ap0)
{
    return float3(
        dot(ap0, float3(2.5214008886, -1.1339957494, -0.3875618568)),
        dot(ap0, float3(-0.2762140616, 1.3725955663, -0.0962823557)),
        dot(ap0, float3(-0.0153202001, -0.1529925618, 1.1683871996)));
}

static const float3 AGX_NEUTRAL_WEIGHTS = float3(
    0.2120053547549465,
    0.3921825078090138,
    0.3958121374360396);

float3 agxInset(float3 color)
{
    float neutral = dot(color, AGX_NEUTRAL_WEIGHTS);
    return lerp(color, neutral.xxx, parameters.agxGamutCompression);
}

float3 agxOutset(float3 color)
{
    float neutral = dot(color, AGX_NEUTRAL_WEIGHTS);
    return (color - parameters.agxGamutCompression * neutral.xxx)
        / (1.0 - parameters.agxGamutCompression);
}

float agxCurveComponent(float value)
{
    bool toe = value <= parameters.agxInputPivot;
    float power = toe ? parameters.agxToePower : parameters.agxShoulderPower;
    float coefficient = toe ? parameters.agxToeA : parameters.agxShoulderA;
    float distance = value - parameters.agxInputPivot;
    return parameters.agxOutputPivot
        + parameters.agxPivotSlope * distance
            * pow(1.0 + coefficient * pow(abs(distance), power), -1.0 / power);
}

float3 agxCurve(float3 value)
{
    return float3(
        agxCurveComponent(value.x),
        agxCurveComponent(value.y),
        agxCurveComponent(value.z));
}

float3 agxS2O3(float3 linearRec709)
{
    // The original AgX-S2O3 OCIO chain applies RangeTransform(minIn=0,
    // minOut=0) before the inset. AP0-to-Rec.709 conversion can produce valid
    // negative wide-gamut components, so enforce that input-domain contract
    // here before taking log2.
    float3 inset = agxInset(max(linearRec709, 0.0));
    float3 normalizedLog = clamp(
        (log2(inset) - parameters.agxMinimumLog2)
            * parameters.agxInverseDynamicRange,
        0.0,
        1.0);
    return agxOutset(agxCurve(normalizedLog));
}

bool hasPositiveInfinity(float3 value)
{
    return (isinf(value.x) && value.x > 0.0)
        || (isinf(value.y) && value.y > 0.0)
        || (isinf(value.z) && value.z > 0.0);
}

bool hasNegativeInfinity(float3 value)
{
    return (isinf(value.x) && value.x < 0.0)
        || (isinf(value.y) && value.y < 0.0)
        || (isinf(value.z) && value.z < 0.0);
}

float3 anomalyColor(bool positiveInfinity, bool negativeInfinity)
{
    if (parameters.showAnomalies != 0)
    {
        if (positiveInfinity && negativeInfinity) return float3(1.0, 0.35, 0.0);
        if (positiveInfinity) return float3(1.0, 1.0, 0.0);
        return float3(0.0, 1.0, 1.0);
    }
    if (positiveInfinity && negativeInfinity) return float3(1.0, 0.0, 1.0);
    if (positiveInfinity) return float3(1.0, 1.0, 1.0);
    return float3(0.0, 0.0, 0.0);
}

float3 prepareOutput(float3 source, float3 mapped)
{
    if (any(isnan(source))) return float3(1.0, 0.0, 1.0);
    bool positiveInfinity = hasPositiveInfinity(source);
    bool negativeInfinity = hasNegativeInfinity(source);
    if (positiveInfinity || negativeInfinity)
        return anomalyColor(positiveInfinity, negativeInfinity);

    if (any(isnan(mapped))) return float3(1.0, 0.0, 1.0);
    positiveInfinity = hasPositiveInfinity(mapped);
    negativeInfinity = hasNegativeInfinity(mapped);
    if (positiveInfinity || negativeInfinity)
        return anomalyColor(positiveInfinity, negativeInfinity);

    if (parameters.showAnomalies != 0)
    {
        if (any(mapped < 0.0)) return float3(0.0, 0.25, 1.0);
        if (any(mapped > 1.0)) return float3(1.0, 0.05, 0.0);
    }

    // agxS2O3 already returns display-encoded BT.709; do not apply another OETF.
    return clamp(mapped, 0.0, 1.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= parameters.width || pixel.y >= parameters.height) return;
    float2 uv = (float2(pixel) + 0.5) / float2(parameters.width, parameters.height);
    float3 ap0 = inputTexture.SampleLevel(inputSampler, uv, 0.0).rgb * parameters.exposureMultiplier;
    float3 mapped = agxS2O3(acesAp0ToRec709(ap0));
    outputTexture.Store(pixel, float4(prepareOutput(ap0, mapped), 1.0));
}
