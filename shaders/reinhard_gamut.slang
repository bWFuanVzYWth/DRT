// Experimental AgX-style virtual-gamut transform around a unified SDR/HDR Reinhard curve.
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
    uint oklabReinhardCurve;
    float reinhardCompressionStart;
    float agxMaximumLogCoordinate;
    float agxOutputPeak;
    float reinhardGamutExpansion;
    float reinhardLinearSlope;
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

static const float3 NEUTRAL_WEIGHTS = float3(
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

float3 toExpandedGamutCoordinates(float3 color)
{
    // Moving the virtual primaries outward makes the coordinates of an
    // unchanged Rec.709 color contract toward the neutral axis.
    float neutral = dot(color, NEUTRAL_WEIGHTS);
    return lerp(color, neutral.xxx, parameters.reinhardGamutExpansion);
}

float3 fromExpandedGamutCoordinates(float3 color)
{
    // The neutral projection is invariant, making this the exact inverse of
    // toExpandedGamutCoordinates before the intervening nonlinear curve.
    float neutral = dot(color, NEUTRAL_WEIGHTS);
    return (color - parameters.reinhardGamutExpansion * neutral.xxx)
        / (1.0 - parameters.reinhardGamutExpansion);
}

float3 reinhard(float3 color)
{
    float linearSlope = parameters.reinhardLinearSlope;
    float compressionStart = parameters.reinhardCompressionStart;
    float startOutput = linearSlope * compressionStart;
    float shoulderExtent = parameters.reinhardCurvePeak - startOutput;
    // The shoulder is unused below the join; keep its denominator positive there.
    float3 distance = max(color - compressionStart, 0.0);
    float3 tangentDistance = linearSlope * distance;
    float3 linear = linearSlope * color;
    float3 shoulder = startOutput
        + tangentDistance / (1.0 + tangentDistance / shoulderExtent);
    return float3(
        color.x <= compressionStart ? linear.x : shoulder.x,
        color.y <= compressionStart ? linear.y : shoulder.y,
        color.z <= compressionStart ? linear.z : shoulder.z);
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

float3 protectHsvHue(float3 originalLinear, float3 mappedDisplay)
{
    if (parameters.reinhardHueRetention <= 0.0)
        return mappedDisplay;

    float3 originalHsv = rgbToHsv(encodeSrgb(originalLinear));
    float3 mappedHsv = rgbToHsv(mappedDisplay);
    if (originalHsv.y <= 1.0e-7 || mappedHsv.y <= 1.0e-7)
        return mappedDisplay;

    float hueOffset = originalHsv.x - mappedHsv.x;
    hueOffset -= floor(hueOffset + 0.5);
    mappedHsv.x = frac(
        mappedHsv.x + parameters.reinhardHueRetention * hueOffset);
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
    float3 ap0 = inputTexture.SampleLevel(inputSampler, uv, 0.0).rgb
        * parameters.exposureMultiplier;
    float3 linearRec709 = max(acesAp0ToRec709(ap0), 0.0);
    float3 working = toExpandedGamutCoordinates(linearRec709);
    float3 mappedLinear = fromExpandedGamutCoordinates(reinhard(working));
    float3 mappedDisplay = encodeSrgb(mappedLinear);
    mappedDisplay = protectHsvHue(linearRec709, mappedDisplay);
    outputTexture.Store(pixel, float4(prepareOutput(ap0, mappedDisplay), 1.0));
}
