Texture2D gSrcTex;
Texture2D gDepthTex;

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

cbuffer CB
{
    float2 dir;
    bool betterSlope;
}

static float depthCache[2 * KERNEL_RADIUS + 1];
static float aoCache[2 * KERNEL_RADIUS + 1];

void AddSample(float sampleAO, float sampleDepth, float depthSlope, uint d, inout float AO, inout float weightSum)
{
    // weights from HBAO+
    const float BlurSigma = ((float)KERNEL_RADIUS+1.0) * 0.5;
    const float BlurFalloff = 1.0 / (2.0*BlurSigma*BlurSigma);
    const float Sharpness = 16.0;

    sampleDepth -= depthSlope * d;
    float dz = abs(sampleDepth - depthCache[KERNEL_RADIUS]) * Sharpness;
    // make dz relative to actual depth
    dz = dz * 12.0 / depthCache[KERNEL_RADIUS];
    
    float w = exp2(-float(d * d)*BlurFalloff - dz * dz);
    AO += w * sampleAO;
    weightSum += w;
}

void BlurDirection(inout float AO, inout float weightSum, float slope, int sign)
{
    float depthSlope = slope;
    [unroll]
    for(uint d = 1; d <= KERNEL_RADIUS; d++)
    {
        float sampleAO = aoCache[KERNEL_RADIUS + sign * d];
        float sampleDepth = depthCache[KERNEL_RADIUS + sign * d];
        if (d == 1 && !betterSlope) depthSlope = sampleDepth - depthCache[KERNEL_RADIUS];

        AddSample(sampleAO, sampleDepth, depthSlope, d, AO, weightSum);
    }
}

float4 main(BlurPSIn pIn) : SV_TARGET
{
    float2 gDUV;
    gSrcTex.GetDimensions(gDUV.x, gDUV.y);
    //duv = rcp(duv) * float2(DIR);
    gDUV = float2(1.0 / gDUV.x, 1.0 / gDUV.y) * dir;

    //fill depth cache
    [unroll]
    for(int d = -KERNEL_RADIUS; d <= KERNEL_RADIUS; d++)
    {
        float2 sampleUv = clamp(pIn.texC + d * gDUV, uvMin, uvMax); // manually clamp to guard band
        depthCache[KERNEL_RADIUS + d] = gDepthTex.Sample(gSampler, sampleUv).x;
        aoCache[KERNEL_RADIUS + d] = gSrcTex.Sample(gSampler, sampleUv).x;
    }
    
    // initial weight (of center sample)
    float AO = aoCache[KERNEL_RADIUS];
    float weightSum = 1.0;

    // compute ao and weights for positive and negative direction
    float slopeLeft = depthCache[KERNEL_RADIUS] - depthCache[KERNEL_RADIUS - 1];
    float slopeRight = depthCache[KERNEL_RADIUS + 1] - depthCache[KERNEL_RADIUS];
    float minSlope = abs(slopeLeft) < abs(slopeRight) ? slopeLeft : slopeRight;
    
    BlurDirection(AO, weightSum, minSlope, 1);
    BlurDirection(AO, weightSum, -minSlope, -1);

    return AO / weightSum;
}
