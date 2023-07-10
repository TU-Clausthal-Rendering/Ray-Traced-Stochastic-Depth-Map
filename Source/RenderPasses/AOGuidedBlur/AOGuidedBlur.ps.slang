Texture2D<float> gSrcTex;
Texture2D<float> gDepthTex;
//Texture2D<float> gImportanceTex;

Texture2D<float> gBrightTex;
Texture2D<float> gDarkTex;

SamplerState gSampler;

struct BlurPSIn
{
    float2 texC : TEXCOORD;
    float4 pos : SV_POSITION;
};

// scissor rectangle
cbuffer ScissorCB
{
    float2 uvMin;
    float2 uvMax;
}

cbuffer Direction
{
    float2 dir;
}

// Simple Gauss-Kernel. Normalization is not included and must be
// done by dividing through the weight sum.
float kernel(float offset, float variance)
{
    return exp(-0.5 * offset * offset / variance);
}

float getDepth(float2 uv)
{
    return gDepthTex.Sample(gSampler, clamp(uv, uvMin, uvMax)).r;
}

float getColor(float2 uv)
{
    return gSrcTex.Sample(gSampler, clamp(uv, uvMin, uvMax)).r;
}

float getBright(float2 uv)
{
    return gBrightTex.Sample(gSampler, clamp(uv, uvMin, uvMax)).r;
}

float getDark(float2 uv)
{
    return gDarkTex.Sample(gSampler, clamp(uv, uvMin, uvMax)).r;
}

float getImportance(float2 uv)
{
    return max(getBright(uv) - getDark(uv), 0.001);
    //return gImportanceTex.Sample(gSampler, clamp(uv, uvMin, uvMax)).r;
}

// returns the local average via wave intrinsics
float LocalAverage(float v)
{
    return 0.25 * (QuadReadLaneAt(v, 0) + QuadReadLaneAt(v, 1) + QuadReadLaneAt(v, 2) + QuadReadLaneAt(v, 3));
}

float WeightedLocalAverage(float v, float4 w)
{
    return w[0] * QuadReadLaneAt(v, 0) + w[1] * QuadReadLaneAt(v, 1) + w[2] * QuadReadLaneAt(v, 2) + w[3] * QuadReadLaneAt(v, 3);
}

float4 LocalDepthWeights(float localDepth)
{
    float4 w;
    [unroll]
    for (uint i = 0; i < 4; ++i)
    {
        w[i] = saturate(abs(QuadReadLaneAt(localDepth, i) / localDepth - 1.0)); // weight in [0, 1] with 0 = equal
        //w[i] = 1.0 - w[i];
        w[i] = kernel(w[i], 0.001);
    }
    // normalize
    return w / (w.x + w.y + w.z + w.w);
}

float main(BlurPSIn pIn) : SV_TARGET
{
    float pixelSum = 0.0;
    float weightSum = 0.0;
    float localDepth = max(getDepth(pIn.texC), 1.401298e-45);
   
    // calc single step in uv coordinates
    float2 uvStep;
    gSrcTex.GetDimensions(uvStep.x, uvStep.y);
    uvStep = float2(1.0 / uvStep.x, 1.0 / uvStep.y) * dir;

    float2 uv = pIn.texC - uvStep * KERNEL_RADIUS;
    for (int it = 0; it <= KERNEL_RADIUS * 2; ++it)
    {
        // calculate weights
        //float spatialWeight = 1.0;
        float spatialWeight = kernel(KERNEL_RADIUS - it, 10.0);
        //float importanceWeight = kernel(getImportance(uv), 0.028);
        float importanceWeight = 1.0 - getImportance(uv);
        //float importanceWeight = 1.0;
        float relativeDepth = saturate(abs(getDepth(uv) / localDepth - 1.0)); // 0.0 = same depth
        float depthWeight = kernel(relativeDepth, 0.001);
        //float depthWeight = kernel(relativeDepth, 0.1);
        //float depthWeight = 1.0;

        // update sum
        float w = spatialWeight * importanceWeight * depthWeight;
        weightSum += w;
        pixelSum += w * getColor(uv);
        
        uv += uvStep;
    }
    
    pixelSum /= max(weightSum, 1e-4);
    if(weightSum < 1e-4)
        pixelSum = getColor(pIn.texC); // fallback color when weight is too small
    
    //if (CLAMP_RESULTS) pixelSum = clamp(pixelSum, getDark(pIn.texC), getBright(pIn.texC));

    if(dir.y != 0) // last iteration
    {
        if (ENHANCE_CONTRAST)
        {
            // contrast enhancement
            float c = pixelSum;
            float4 w = LocalDepthWeights(localDepth);
            //w = 0.25;
            float cAvg = WeightedLocalAverage(c, w);

            float cBright = getBright(pIn.texC);
            float cBrightAvg = WeightedLocalAverage(cBright, w);
            float brightDev = cBright - cBrightAvg;

            float cDark = getDark(pIn.texC);
            float cDarkAvg = WeightedLocalAverage(cDark, w);
            float darkDev = cDark - cDarkAvg;

            //pixelSum = cAvg + brightDev;
            pixelSum = cAvg + min(brightDev, 0.0) * 2.0;
            //pixelSum = cAvg + min(brightDev, darkDev);
        }
        
        // clamp between bright and dark image
        if (CLAMP_RESULTS)
            pixelSum = clamp(pixelSum, getDark(pIn.texC), getBright(pIn.texC));
    }
    
    return pixelSum;
}
