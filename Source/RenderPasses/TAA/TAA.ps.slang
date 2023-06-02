/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
import Utils.Color.ColorHelpers;

cbuffer PerFrameCB
{
    float gAlpha;
    float gColorBoxSigma;
    bool gAntiFlicker;
    bool gUseDepth;
};

Texture2D gTexColor;
Texture2D gTexMotionVec;
Texture2D gTexPrevColor;
SamplerState gSampler;

Texture2D<float> gDepth;
Texture2D<float> gPrevDepth;

float RelativeDepth(float reference, float value)
{
    return abs(1.0 - value / reference);
}

float3 bilinearSampleWithDepthWeights(Texture2D colorTex, Texture2D<float> depthTex, float2 samplePos, float2 texDim, float depthRef)
{
    int2 p00 = floor(samplePos - 0.5); // top left pixel
    //float2 w = samplePos - float2(p00) - 0.5; // weight for x and y interpolation
    return colorTex.SampleLevel(gSampler, samplePos / texDim, 0.0).rgb;
    // fetch colors and depths
    float3 c00 = colorTex[p00].rgb;
    float3 c10 = colorTex[p00 + int2(1, 0)].rgb;
    float3 c01 = colorTex[p00 + int2(0, 1)].rgb;
    float3 c11 = colorTex[p00 + int2(1, 1)].rgb;

    float d00 = depthTex[p00];
    float d10 = depthTex[p00 + int2(1, 0)];
    float d01 = depthTex[p00 + int2(0, 1)];
    float d11 = depthTex[p00 + int2(1, 1)];


    return
    c00 * saturate((depthRef - d10) / (d00 - d10)) * saturate((depthRef - d01) / (d00 - d01)) +
    c10 * saturate((depthRef - d00) / (d10 - d00)) * saturate((depthRef - d11) / (d10 - d11)) +
    c01 * saturate((depthRef - d11) / (d01 - d11)) * saturate((depthRef - d00) / (d01 - d00)) +
    c11 * saturate((depthRef - d01) / (d11 - d01)) * saturate((depthRef - d10) / (d11 - d10));
}

// Catmull-Rom filtering code from http://vec3.ca/bicubic-filtering-in-fewer-taps/
float3 bicubicSampleCatmullRom(Texture2D tex, SamplerState samp, float2 samplePos, float2 texDim)
{
    float2 invTextureSize = 1.0 / texDim;
    float2 tc = floor(samplePos - 0.5f) + 0.5f;
    float2 f = samplePos - tc;
    float2 f2 = f * f;
    float2 f3 = f2 * f;

    float2 w0 = f2 - 0.5f * (f3 + f);
    float2 w1 = 1.5f * f3 - 2.5f * f2 + 1.f;
    float2 w3 = 0.5f * (f3 - f2);
    float2 w2 = 1 - w0 - w1 - w3;

    float2 w12 = w1 + w2;

    float2 tc0 = (tc - 1.f) * invTextureSize;
    float2 tc12 = (tc + w2 / w12) * invTextureSize;
    float2 tc3 = (tc + 2.f) * invTextureSize;

    float3 result =
        tex.SampleLevel(samp, float2(tc0.x,  tc0.y), 0.f).rgb  * (w0.x  * w0.y) +
        tex.SampleLevel(samp, float2(tc0.x,  tc12.y), 0.f).rgb * (w0.x  * w12.y) +
        tex.SampleLevel(samp, float2(tc0.x,  tc3.y), 0.f).rgb  * (w0.x  * w3.y) +
        tex.SampleLevel(samp, float2(tc12.x, tc0.y), 0.f).rgb  * (w12.x * w0.y) +
        tex.SampleLevel(samp, float2(tc12.x, tc12.y), 0.f).rgb * (w12.x * w12.y) +
        tex.SampleLevel(samp, float2(tc12.x, tc3.y), 0.f).rgb  * (w12.x * w3.y) +
        tex.SampleLevel(samp, float2(tc3.x,  tc0.y), 0.f).rgb  * (w3.x  * w0.y) +
        tex.SampleLevel(samp, float2(tc3.x,  tc12.y), 0.f).rgb * (w3.x  * w12.y) +
        tex.SampleLevel(samp, float2(tc3.x,  tc3.y), 0.f).rgb  * (w3.x  * w3.y);

    return result;
}



float4 main(float2 texC : TEXCOORD) : SV_TARGET0
{
    const int2 offset[8] = { int2(-1, -1), int2(-1,  1),
                              int2( 1, -1), int2( 1,  1),
                              int2( 1,  0), int2( 0, -1),
                              int2( 0,  1), int2(-1,  0), };

    uint2 texDim;
    uint levels;
    gTexColor.GetDimensions(0, texDim.x, texDim.y, levels);

    float2 pos = texC * texDim;
    int2 ipos = int2(pos);

    // Fetch the current pixel color and compute the color bounding box
    // Details here: http://www.gdcvault.com/play/1023521/From-the-Lab-Bench-Real
    // and here: http://cwyman.org/papers/siga16_gazeTrackedFoveatedRendering.pdf
    float3 color = gTexColor.Load(int3(ipos, 0)).rgb;
    float depth = gDepth[ipos];
    color = RGBToYCgCo(color);
    float3 colorAvg = color;
    float3 colorVar = color * color;
    [unroll]
    for (int k = 0; k < 8; k++)
    {
        float3 c = gTexColor.Load(int3(ipos + offset[k], 0)).rgb;
        c = RGBToYCgCo(c);
        colorAvg += c;
        colorVar += c * c;
    }

    float oneOverNine = 1.f / 9.f;
    colorAvg *= oneOverNine;
    colorVar *= oneOverNine;

    float3 sigma = sqrt(max(0.f, colorVar - colorAvg * colorAvg));
    float3 colorMin = colorAvg - gColorBoxSigma * sigma;
    float3 colorMax = colorAvg + gColorBoxSigma * sigma;


    float2 motion = gTexMotionVec.Load(int3(ipos, 0)).xy;

    // Find the longest motion vector    
    [unroll]
    for (int a = 0; a < 8; a++)
    {
        float2 m = gTexMotionVec.Load(int3(ipos + offset[a], 0)).rg;
        motion = dot(m, m) > dot(motion, motion) ? m : motion;
    }

    /*float prevDepth = gPrevDepth[int2((texC + motion) * texDim)];
    if (gUseDepth && RelativeDepth(depth, prevDepth) > 0.1)
        return float4(YCgCoToRGB(color), 1.0); // depth differs too much
    */
    // Use motion vector to fetch previous frame color (history)
    float3 history;
    if(gUseDepth)
        history = bilinearSampleWithDepthWeights(gTexPrevColor, gPrevDepth, (texC + motion) * texDim, texDim, depth);
    else
        history = bicubicSampleCatmullRom(gTexPrevColor, gSampler, (texC + motion) * texDim, texDim);

    history = RGBToYCgCo(history);

    float alpha = gAlpha;

    // Anti-flickering, based on Brian Karis talk @Siggraph 2014
    // https://de45xmedrsdbp.cloudfront.net/Resources/files/TemporalAA_small-59732822.pdf
    // Reduce blend factor when history is near clamping
    if (gAntiFlicker)
    {
        float distToClamp = min(abs(colorMin.x - history.x), abs(colorMax.x - history.x));
        alpha = clamp((gAlpha * distToClamp) / (distToClamp + colorMax.x - colorMin.x), 0.f, 1.f);
    }

    history = clamp(history, colorMin, colorMax);
    float3 result = YCgCoToRGB(lerp(history, color, alpha));
    return float4(result, 1.f);
}
