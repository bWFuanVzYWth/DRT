// Oklab DRT developed by this project's authors; drt-bench was its development harness.
// Ported to Slang, SPIR-V, and wgpu on 2026-08-13. Project license is not selected yet.
// Input: scene-linear ACES2065-1 (AP0). Output: display-encoded sRGB.

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

static const float OKLAB_DRT_RGB_HEADROOM = 0.99999;
static const float OKLAB_DRT_MIDDLE_GRAY = 0.18;
static const float3 OKLAB_DRT_RED_ROW = float3(4.0767416621, -3.3077115913, 0.2309699292);
static const float3 OKLAB_DRT_GREEN_ROW = float3(-1.2684380046, 2.6097574011, -0.3413193965);
static const float3 OKLAB_DRT_BLUE_ROW = float3(-0.0041960863, -0.7034186147, 1.7076147010);

float3 encodeSrgb(float3 linearRgb)
{
    bool3 cutoff = linearRgb < 0.0031308;
    float3 higher = 1.055 * pow(linearRgb, 1.0 / 2.4) - 0.055;
    float3 lower = linearRgb * 12.92;
    return float3(
        cutoff.x ? lower.x : higher.x,
        cutoff.y ? lower.y : higher.y,
        cutoff.z ? lower.z : higher.z);
}

float3 rgbToOklab(float3 color)
{
    float l = 0.4122214708 * color.r + 0.5363325363 * color.g + 0.0514459929 * color.b;
    float m = 0.2119034982 * color.r + 0.6806995451 * color.g + 0.1073969566 * color.b;
    float s = 0.0883024619 * color.r + 0.2817188376 * color.g + 0.6299787005 * color.b;
    float3 lms = pow(float3(l, m, s), 1.0 / 3.0);
    return float3(
        0.2104542553 * lms.x + 0.7936177850 * lms.y - 0.0040720468 * lms.z,
        1.9779984951 * lms.x - 2.4285922050 * lms.y + 0.4505937099 * lms.z,
        0.0259040371 * lms.x + 0.7827717662 * lms.y - 0.8086757660 * lms.z);
}

float3 oklabToRgb(float3 color)
{
    float lRoot = color.x + 0.3963377774 * color.y + 0.2158037573 * color.z;
    float mRoot = color.x - 0.1055613458 * color.y - 0.0638541728 * color.z;
    float sRoot = color.x - 0.0894841775 * color.y - 1.2914855480 * color.z;
    float3 lms = float3(lRoot * lRoot * lRoot, mRoot * mRoot * mRoot, sRoot * sRoot * sRoot);
    return float3(
        dot(OKLAB_DRT_RED_ROW, lms),
        dot(OKLAB_DRT_GREEN_ROW, lms),
        dot(OKLAB_DRT_BLUE_ROW, lms));
}

float shoulderCoefficient(float overexposure)
{
    float scale = overexposure / (overexposure - OKLAB_DRT_MIDDLE_GRAY);
    return (scale * scale - 1.0) / OKLAB_DRT_MIDDLE_GRAY;
}

float mapLightness(float lightness, float overexposure)
{
    float brightness = lightness * lightness * lightness;
    if (parameters.oklabReinhardCurve != 0)
    {
        // Match Reinhard-Gamut's linear toe and tangent-continuous shoulder
        // in L^3, then return to Oklab lightness for the existing gamut mapping.
        float linearSlope = parameters.reinhardLinearSlope;
        float compressionStart = parameters.reinhardCompressionStart;
        float mappedBrightness;
        if (brightness <= compressionStart)
        {
            mappedBrightness = linearSlope * brightness;
        }
        else
        {
            float startOutput = linearSlope * compressionStart;
            float shoulderExtent = parameters.reinhardCurvePeak - startOutput;
            float tangentDistance = linearSlope * (brightness - compressionStart);
            mappedBrightness = startOutput
                + tangentDistance / (1.0 + tangentDistance / shoulderExtent);
        }
        return pow(mappedBrightness, 1.0 / 3.0);
    }
    float x = shoulderCoefficient(overexposure) * brightness;
    float inverseRoot = rsqrt(1.0 + x);
    float mappedBrightness = overexposure * x * inverseRoot * inverseRoot / (1.0 + inverseRoot);
    return pow(mappedBrightness, 1.0 / 3.0);
}

float3 rootDirection(float2 hue)
{
    return float3(
        0.3963377774 * hue.x + 0.2158037573 * hue.y,
        -0.1055613458 * hue.x - 0.0638541728 * hue.y,
        -0.0894841775 * hue.x - 1.2914855480 * hue.y);
}

float maxSaturation(float2 hue, float3 direction)
{
    float k0;
    float k1;
    float k2;
    float k3;
    float k4;
    float3 rgbRow;
    if (-1.88170328 * hue.x - 0.80936493 * hue.y > 1.0)
    {
        k0 = 1.19086277; k1 = 1.76576728; k2 = 0.59662641; k3 = 0.75515197; k4 = 0.56771245;
        rgbRow = OKLAB_DRT_RED_ROW;
    }
    else if (1.81444104 * hue.x - 1.19445276 * hue.y > 1.0)
    {
        k0 = 0.73956515; k1 = -0.45954404; k2 = 0.08285427; k3 = 0.12541070; k4 = 0.14503204;
        rgbRow = OKLAB_DRT_GREEN_ROW;
    }
    else
    {
        k0 = 1.35733652; k1 = -0.00915799; k2 = -1.15130210; k3 = -0.50559606; k4 = 0.00692167;
        rgbRow = OKLAB_DRT_BLUE_ROW;
    }

    float saturation = k0 + k1 * hue.x + k2 * hue.y + k3 * hue.x * hue.x + k4 * hue.x * hue.y;
    float3 roots = 1.0 + saturation * direction;
    float3 lms = roots * roots * roots;
    float3 firstLms = 3.0 * direction * roots * roots;
    float3 secondLms = 6.0 * direction * direction * roots;
    float f = dot(rgbRow, lms);
    float f1 = dot(rgbRow, firstLms);
    float f2 = dot(rgbRow, secondLms);
    return saturation - f * f1 / (f1 * f1 - 0.5 * f * f2);
}

float connectedSaturation(float2 hue, float saturation)
{
    const float2 blueNotchAxis = float2(-0.10362546, -0.99461639);
    float alignment = max(dot(hue, blueNotchAxis), 0.0);
    float alignment2 = alignment * alignment;
    float alignment4 = alignment2 * alignment2;
    float alignment8 = alignment4 * alignment4;
    float alignment16 = alignment8 * alignment8;
    float alignment32 = alignment16 * alignment16;
    float alignment64 = alignment32 * alignment32;
    float alignment128 = alignment64 * alignment64;
    float alignment256 = alignment128 * alignment128;
    return min(saturation, 0.57 + (1.0 - alignment256));
}

float cuspLightness(float saturation, float3 direction)
{
    float3 roots = 1.0 + saturation * direction;
    float3 lms = roots * roots * roots;
    float3 rgb = float3(
        dot(OKLAB_DRT_RED_ROW, lms),
        dot(OKLAB_DRT_GREEN_ROW, lms),
        dot(OKLAB_DRT_BLUE_ROW, lms));
    return pow(1.0 / max(rgb.r, max(rgb.g, rgb.b)), 1.0 / 3.0);
}

float refineUpperChroma(float chroma, float lightness, float3 direction)
{
    float3 roots = lightness + chroma * direction;
    float3 lms = roots * roots * roots;
    float3 firstLms = 3.0 * direction * roots * roots;
    float3 secondLms = 6.0 * direction * direction * roots;
    float3 rgb = float3(dot(OKLAB_DRT_RED_ROW, lms), dot(OKLAB_DRT_GREEN_ROW, lms), dot(OKLAB_DRT_BLUE_ROW, lms));
    float3 firstRgb = float3(dot(OKLAB_DRT_RED_ROW, firstLms), dot(OKLAB_DRT_GREEN_ROW, firstLms), dot(OKLAB_DRT_BLUE_ROW, firstLms));
    float3 secondRgb = float3(dot(OKLAB_DRT_RED_ROW, secondLms), dot(OKLAB_DRT_GREEN_ROW, secondLms), dot(OKLAB_DRT_BLUE_ROW, secondLms));
    float3 f = rgb - 1.0;
    float3 denominator = firstRgb * firstRgb - 0.5 * f * secondRgb;
    float3 reciprocalStep = firstRgb / denominator;
    float3 step = -f * reciprocalStep;
    step = float3(
        reciprocalStep.x >= 0.0 ? step.x : 1.0e20,
        reciprocalStep.y >= 0.0 ? step.y : 1.0e20,
        reciprocalStep.z >= 0.0 ? step.z : 1.0e20);
    return chroma + min(step.r, min(step.g, step.b));
}

float softMin(float value, float limit, float power)
{
    if (value <= 0.0 || limit <= 0.0) return 0.0;
    float lower = min(value, limit);
    float higher = max(value, limit);
    float ratio = lower / higher;
    return lower * pow(1.0 + pow(ratio, power), -1.0 / power);
}

float softMin4(float value, float limit)
{
    if (value <= 0.0 || limit <= 0.0) return 0.0;
    float lower = min(value, limit);
    float higher = max(value, limit);
    float ratio = lower / higher;
    float ratio2 = ratio * ratio;
    float root = sqrt(1.0 + ratio2 * ratio2);
    return lower * rsqrt(root);
}

float saturationCap(float lightness, float maximumSaturation, float3 direction)
{
    if (lightness <= 0.0) return maximumSaturation;
    if (lightness >= 1.0) return 0.0;
    float cusp = cuspLightness(maximumSaturation, direction);
    float blackChroma = lightness * maximumSaturation;
    float whiteChroma = cusp * maximumSaturation * (1.0 - lightness) / (1.0 - cusp);
    whiteChroma = refineUpperChroma(whiteChroma, lightness, direction);
    float t = clamp((lightness - cusp) / (1.0 - cusp), 0.0, 1.0);
    float shoulder = t * (1.0 - t);
    whiteChroma *= 1.0 - 0.0035 * 16.0 * shoulder * shoulder;
    float roundedChroma = softMin4(blackChroma, whiteChroma);
    return max(roundedChroma / lightness, 0.0);
}

float chromaRetention(float lightness)
{
    float lightness2 = lightness * lightness;
    float lightness4 = lightness2 * lightness2;
    float lightness8 = lightness4 * lightness4;
    return 1.0 - lightness8 * lightness4;
}

float roundingPower(float lightness)
{
    float endpointDistance = lightness * (1.0 - lightness);
    return 32.0 - 256.0 * endpointDistance * endpointDistance;
}

float3 mapLinearRgb(float3 color, float overexposure)
{
    float3 oklab = rgbToOklab(color);
    if (oklab.x <= 0.0) return 0.0;
    float outputLightness = mapLightness(oklab.x, overexposure);
    float inputChroma = length(oklab.yz);
    if (inputChroma <= 1.0e-8)
        return OKLAB_DRT_RGB_HEADROOM * oklabToRgb(float3(outputLightness, 0.0, 0.0));

    float2 hue = oklab.yz / inputChroma;
    float3 direction = rootDirection(hue);
    float inputSaturation = inputChroma / oklab.x;
    float maximumSaturation = connectedSaturation(hue, maxSaturation(hue, direction));
    float desiredSaturation = inputSaturation * chromaRetention(outputLightness);
    float cap = saturationCap(outputLightness, maximumSaturation, direction);
    float outputSaturation = softMin(desiredSaturation, cap, roundingPower(outputLightness));
    return OKLAB_DRT_RGB_HEADROOM * oklabToRgb(float3(
        outputLightness, outputLightness * outputSaturation * hue));
}

float3 acesAp0ToRec709(float3 ap0)
{
    return float3(
        dot(ap0, float3(2.5214008886, -1.1339957494, -0.3875618568)),
        dot(ap0, float3(-0.2762140616, 1.3725955663, -0.0962823557)),
        dot(ap0, float3(-0.0153202001, -0.1529925618, 1.1683871996)));
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
    // Magenta is reserved for undefined values in both modes.
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
        // Distinct diagnostic classes for finite range errors.
        if (any(mapped < 0.0)) return float3(0.0, 0.25, 1.0);
        if (any(mapped > 1.0)) return float3(1.0, 0.05, 0.0);
    }

    return clamp(encodeSrgb(clamp(mapped, 0.0, 1.0)), 0.0, 1.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= parameters.width || pixel.y >= parameters.height) return;
    float2 uv = (float2(pixel) + 0.5) / float2(parameters.width, parameters.height);
    float3 ap0 = inputTexture.SampleLevel(inputSampler, uv, 0.0).rgb * parameters.exposureMultiplier;
    float3 mapped = mapLinearRgb(acesAp0ToRec709(ap0), parameters.overexposure);
    outputTexture.Store(pixel, float4(prepareOutput(ap0, mapped), 1.0));
}
