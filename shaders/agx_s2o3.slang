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

float3 agxInset(float3 color)
{
    // The source GLSL mat3 constructor is column-major. These explicit rows
    // preserve the actual result of `agx_mat * color` across shader languages.
    return float3(
        dot(color, float3(0.8424010709504686, 0.07843650156180276, 0.0791624274877287)),
        dot(color, float3(0.04240107095046854, 0.8784365015618028, 0.0791624274877287)),
        dot(color, float3(0.04240107095046854, 0.07843650156180276, 0.8791624274877287)));
}

float3 agxOutset(float3 color)
{
    return float3(
        dot(color, float3(1.1969986613119143, -0.09804562695225345, -0.09895303435966087)),
        dot(color, float3(-0.053001338688085674, 1.1519543730477466, -0.09895303435966087)),
        dot(color, float3(-0.053001338688085674, -0.09804562695225345, 1.151046965640339)));
}

float3 agxCurve(float3 value)
{
    const float threshold = 0.6060606060606061;
    const float aUp = 69.86278913545539;
    const float aDown = 59.507875;
    const float bUp = 13.0 / 4.0;
    const float bDown = 3.0;
    const float cUp = -4.0 / 13.0;
    const float cDown = -1.0 / 3.0;

    float3 mask = float3(
        value.x <= threshold ? 1.0 : 0.0,
        value.y <= threshold ? 1.0 : 0.0,
        value.z <= threshold ? 1.0 : 0.0);
    float3 a = aUp + (aDown - aUp) * mask;
    float3 b = bUp + (bDown - bUp) * mask;
    float3 c = cUp + (cDown - cUp) * mask;
    return 0.5
        + ((-2.0 * threshold) + 2.0 * value)
            * pow(1.0 + a * pow(abs(value - threshold), b), c);
}

float3 agxS2O3(float3 linearRec709)
{
    const float minimumEv = -12.473931188332413;
    const float maximumEv = 4.026068811667588;
    const float inverseDynamicRange = 1.0 / (maximumEv - minimumEv);

    // The original AgX-S2O3 OCIO chain applies RangeTransform(minIn=0,
    // minOut=0) before the inset. AP0-to-Rec.709 conversion can produce valid
    // negative wide-gamut components, so enforce that input-domain contract
    // here before taking log2.
    float3 inset = agxInset(max(linearRec709, 0.0));
    float3 normalizedLog = clamp(
        log2(inset) * inverseDynamicRange - minimumEv * inverseDynamicRange,
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
