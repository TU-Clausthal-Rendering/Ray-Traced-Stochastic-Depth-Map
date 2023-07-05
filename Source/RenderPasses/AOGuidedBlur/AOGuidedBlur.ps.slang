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
        float spatialWeight = 1.0;
        //float importanceWeight = kernel(getImportance(uv), 0.028);
        float importanceWeight = 1.0 - getImportance(uv);
        float relativeDepth = saturate(abs(getDepth(uv) / localDepth - 1.0)); // 0.0 = same depth
        float depthWeight = kernel(relativeDepth, 0.001);

        // update sum
        float w = spatialWeight * importanceWeight * depthWeight;
        weightSum += w;
        pixelSum += w * getColor(uv);
        
        uv += uvStep;
    }
    
    pixelSum /= weightSum;
    
    // clamp between bright and dark image
    pixelSum = clamp(pixelSum, getDark(pIn.texC), getBright(pIn.texC));
    
    return pixelSum;
}
