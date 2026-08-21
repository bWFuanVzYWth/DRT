// Display reference with no tone mapping.
// Input: scene-linear ACES2065-1 (AP0).
// Output: display-encoded extended sRGB (1.0 is SDR reference white).

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
    float oklabWhiteHueCompression;
    float oklabHueCompressionOnset;
    float agxMaximumLogCoordinate;
    float agxOutputPeak;
    float reinhardGamutExpansion;
    float reinhardMiddleGray;
    float reinhardOutputPeak;
    float reinhardHueRetention;
    float reinhardCurvePeak;
    uint padding3;
    uint padding4;
    uint padding5;
};

[[vk::binding(0, 0)]] Texture2D<float4> inputTexture;
[[vk::binding(1, 0)]] SamplerState inputSampler;
[[vk::binding(2, 0)]] [vk::image_format("rgba16f")] WTexture2D<float4> outputTexture;
[[vk::binding(3, 0)]] ConstantBuffer<DrtParameters> parameters;

float3 acesAp0ToRec709(float3 ap0)
{
    return float3(
        dot(ap0, float3(2.5214008886, -1.1339957494, -0.3875618568)),
        dot(ap0, float3(-0.2762140616, 1.3725955663, -0.0962823557)),
        dot(ap0, float3(-0.0153202001, -0.1529925618, 1.1683871996)));
}

float3 encodeSrgb(float3 linearRgb)
{
    bool3 cutoff = linearRgb < 0.0031308;
    float3 higher = 1.055 * pow(max(linearRgb, 0.0), 1.0 / 2.4) - 0.055;
    float3 lower = linearRgb * 12.92;
    return float3(
        cutoff.x ? lower.x : higher.x,
        cutoff.y ? lower.y : higher.y,
        cutoff.z ? lower.z : higher.z);
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

    float encodedOutputPeak = encodeSrgb(parameters.reinhardOutputPeak.xxx).x;
    if (parameters.showAnomalies != 0)
    {
        if (any(mapped < 0.0)) return float3(0.0, 0.25, 1.0);
        if (any(mapped > encodedOutputPeak)) return float3(1.0, 0.05, 0.0);
    }
    return clamp(mapped, 0.0, encodedOutputPeak);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= parameters.width || pixel.y >= parameters.height) return;
    float2 uv = (float2(pixel) + 0.5) / float2(parameters.width, parameters.height);
    float3 ap0 = inputTexture.SampleLevel(inputSampler, uv, 0.0).rgb * parameters.exposureMultiplier;
    float3 mapped = encodeSrgb(acesAp0ToRec709(ap0));
    outputTexture.Store(pixel, float4(prepareOutput(ap0, mapped), 1.0));
}
