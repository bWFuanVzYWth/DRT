// AgX-S2O3 followed by an optional HSV hue repair.
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

static const float3 AGX_NEUTRAL_WEIGHTS = float3(
    0.2120053547549465,
    0.3921825078090138,
    0.3958121374360396);

float3 acesAp0ToRec709(float3 ap0)
{
    return float3(
        dot(ap0, float3(2.5214008886, -1.1339957494, -0.3875618568)),
        dot(ap0, float3(-0.2762140616, 1.3725955663, -0.0962823557)),
        dot(ap0, float3(-0.0153202001, -0.1529925618, 1.1683871996)));
}

float3 rgbToHsv(float3 color)
{
    float maximum = max(color.r, max(color.g, color.b));
    float minimum = min(color.r, min(color.g, color.b));
    float chroma = maximum - minimum;
    float hue = 0.0;
    if (chroma > 1.0e-7)
    {
        if (maximum == color.r)
            hue = (color.g - color.b) / chroma;
        else if (maximum == color.g)
            hue = (color.b - color.r) / chroma + 2.0;
        else
            hue = (color.r - color.g) / chroma + 4.0;
        hue = frac(hue / 6.0);
    }
    float saturation = maximum > 1.0e-7 ? chroma / maximum : 0.0;
    return float3(hue, saturation, maximum);
}

float3 hsvToRgb(float3 hsv)
{
    float3 primary = clamp(
        abs(frac(hsv.x + float3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0,
        0.0,
        1.0);
    return hsv.z * lerp(1.0.xxx, primary, hsv.y);
}

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
    float3 inset = agxInset(linearRec709);
    float3 normalizedLog = clamp(
        (log2(inset) - parameters.agxMinimumLog2)
            * parameters.agxInverseDynamicRange,
        0.0,
        1.0);
    return agxOutset(agxCurve(normalizedLog));
}

float compressHsvSaturation(float saturation, float onset)
{
    if (saturation <= onset) return saturation;
    float headroom = 1.0 - onset;
    float excess = saturation - onset;
    return onset + headroom * excess / (headroom + excess);
}

float3 adjustHsv(float3 originalLinear, float3 mappedDisplay)
{
    // Compare both hues in the same pure-2.2 signal domain used by AgX Base.
    float3 originalDisplay = pow(originalLinear, 1.0 / 2.2);
    float3 originalHsv = rgbToHsv(originalDisplay);
    float3 mappedHsv = rgbToHsv(mappedDisplay);
    float normalizedValue = clamp(mappedHsv.z, 0.0, 1.0);

    if (originalHsv.y > 1.0e-7 && mappedHsv.y > 1.0e-7)
    {
        float retention = lerp(
            parameters.agxBlackHueRetention,
            parameters.agxWhiteHueRetention,
            normalizedValue);
        float hueOffset = originalHsv.x - mappedHsv.x;
        hueOffset -= floor(hueOffset + 0.5);
        mappedHsv.x = frac(mappedHsv.x + retention * hueOffset);
    }

    float gamutOnset = lerp(
        parameters.agxBlackGamutOnset,
        1.0,
        normalizedValue);
    mappedHsv.y = compressHsvSaturation(mappedHsv.y, gamutOnset);
    return hsvToRgb(mappedHsv);
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
    float3 originalLinear = max(acesAp0ToRec709(ap0), 0.0);
    float3 mapped = adjustHsv(originalLinear, agxS2O3(originalLinear));
    outputTexture.Store(pixel, float4(prepareOutput(ap0, mapped), 1.0));
}
