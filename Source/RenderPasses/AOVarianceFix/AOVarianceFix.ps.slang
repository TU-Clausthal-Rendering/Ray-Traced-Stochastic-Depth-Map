Texture2D<float> gDepthTex;
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

float2 getBrightDark(float2 uv)
{
    return float2(gBrightTex.Sample(gSampler, clamp(uv, uvMin, uvMax)).r, gDarkTex.Sample(gSampler, clamp(uv, uvMin, uvMax)).r);
}

float main(BlurPSIn pIn) : SV_TARGET
{
    float localDepth = max(getDepth(pIn.texC), 1.401298e-45);
    float2 local = getBrightDark(pIn.texC); // local color
    float2 means = local;
    float weightSum = 1.0;
    
    // calc single step in uv coordinates
    float2 uvStep;
    gBrightTex.GetDimensions(uvStep.x, uvStep.y);
    uvStep = float2(1.0 / uvStep.x, 1.0 / uvStep.y);


    float2 offsets[] = { float2(-1, 0), float2(0, -1), float2(1, 0), float2(0, 1) };
    [unroll] for (int it = 0; it < 4; ++it)
    {
        float2 uv = pIn.texC + offsets[it] * uvStep;
        
        // calculate weights
        float spatialWeight = kernel(offsets[it].x, 10.0) * kernel(offsets[it].y, 10.0);;
        float importanceWeight = 1.0;
        float relativeDepth = saturate(abs(getDepth(uv) / localDepth - 1.0)); // 0.0 = same depth
        float depthWeight = kernel(relativeDepth, 0.001);
        
        // update sum
        float w = spatialWeight * importanceWeight * depthWeight;
        weightSum += w;
        means += w * getBrightDark(uv);
    }

    /*float2 offsets[] = { float2(-1.5, -0.5), float2(0.5, -1.5), float2(1.5, 0.5), float2(-0.5, 1.5) };
    [unroll]
    for (int it = 0; it < 4; ++it)
    {
        float2 gatherUV = pIn.texC + offsets[it] * uvStep;
        
        // calculate weights
        float spatialWeight = kernel(offsets[it].x, 10.0) * kernel(offsets[it].y, 10.0);;
        float importanceWeight = 1.0;
        float relativeDepth = saturate(abs(getDepth(uv) / localDepth - 1.0)); // 0.0 = same depth
        float depthWeight = kernel(relativeDepth, 0.001);
        
        // update sum
        float w = spatialWeight * importanceWeight * depthWeight;
        weightSum += w;
        means += w * float2(getBright(uv), getDark(uv));
    }*/
    
    means /= max(weightSum, 1e-4);
    if (any(means < 1e-4))
        means = local; // fallback 

    float2 dev = local - means;

    float2 adev = abs(dev); // for weights
    adev.y = max(adev.y, 0.01); // prevent division by zero if both are zero

    float2 w = adev.yx / (adev.x + adev.y); // normalized weight

    float ao = dot(local, w);
    
    return ao;
}
