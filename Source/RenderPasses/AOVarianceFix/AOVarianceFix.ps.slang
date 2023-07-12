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

float main(BlurPSIn pIn) : SV_TARGET
{
    float2 means = 0.0;
    float weightSum = 0.0;
    float localDepth = max(getDepth(pIn.texC), 1.401298e-45);
    float2 local = float2(getBright(pIn.texC), getDark(pIn.texC)); // local color
    
    // calc single step in uv coordinates
    float2 uvStep;
    gBrightTex.GetDimensions(uvStep.x, uvStep.y);
    uvStep = float2(1.0 / uvStep.x, 1.0 / uvStep.y);

    float2 offsets[] = { float2(-1, 0), float2(0, -1), float2(0, 0), float2(1, 0), float2(0, 1) };

    [unroll] for (int it = 0; it < 5; ++it)
    {
        float2 uv = pIn.texC + offsets[it] * uvStep;
        
        // calculate weights
        //float spatialWeight = 1.0;
        float spatialWeight = kernel(offsets[it].x + offsets[it].y, 10.0);
        //float importanceWeight = kernel(getImportance(uv), 0.028);
        //float importanceWeight = 1.0 - getImportance(uv);
        float importanceWeight = 1.0;
        float relativeDepth = saturate(abs(getDepth(uv) / localDepth - 1.0)); // 0.0 = same depth
        float depthWeight = kernel(relativeDepth, 0.001);
        
        // update sum
        float w = spatialWeight * importanceWeight * depthWeight;
        weightSum += w;
        means += w * float2(getBright(uv), getDark(uv));
        
    }
    
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
