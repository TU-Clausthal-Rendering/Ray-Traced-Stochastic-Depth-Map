Texture2D gSrcTex;
Texture2D<float> gDepthTex;
//Texture2D<float> gImportanceTex;

Texture2D<float2> gBrightDarkTex;

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

// not used anymore
#define IMPORTANCE_EXPONENT 1.0


// optimized city
//#define DEPTH_VARIANCE  0.0049
//#define SPATIAL_VARIANCE 10.4
//#define DEV_EXPONENT 0.32
//#define DARK_EPSILON 0.76
//#define ENHANCE_CONTRAST 0.40

// optimized park
#define DEPTH_VARIANCE  0.0003
#define SPATIAL_VARIANCE 7.4
#define DEV_EXPONENT 0.3
#define DARK_EPSILON 0.38
#define ENHANCE_CONTRAST 0.82


// original params w/o optimze
//#define DEPTH_VARIANCE 0.001
//#define SPATIAL_VARIANCE 16.4
//#define DEV_EXPONENT 1.0
//#define DARK_EPSILON 0.01
//#define ENHANCE_CONTRAST 1.0

#define BRIGHT(v) (v.x)
#define DARK(v) (v.y)

#define OUTPUT_RESULT 0
#define OUTPUT_BRIGHT_MEAN 1
#define OUTPUT_DARK_MEAN 2
#define OUTPUT_BRIGHT_DEV 3
#define OUTPUT_DARK_DEV 4
#define OUTPUT_BRIGHT_LOCAL_DEV 5
#define OUTPUT_DARK_LOCAL_DEV 6
#define OUTPUT_BRIGHT 7
#define OUTPUT_DARK 8

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

float4 getColor(float2 uv)
{
    return gSrcTex.Sample(gSampler, clamp(uv, uvMin, uvMax));
}

float2 getOriginalColor(float2 uv)
{
    return gBrightDarkTex.Sample(gSampler, clamp(uv, uvMin, uvMax));
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

float4 main(BlurPSIn pIn) : SV_TARGET
{
    float2 means = 0.0;
    float2 meansSq = 0.0;
    float weightSum = 0.0;
    float localDepth = max(getDepth(pIn.texC), 1.401298e-45);
    float2 local = getColor(pIn.texC).xy;
       
    // calc single step in uv coordinates
    float2 uvStep;
    gSrcTex.GetDimensions(uvStep.x, uvStep.y);
    uvStep = float2(1.0 / uvStep.x, 1.0 / uvStep.y) * dir;

    float2 uv = pIn.texC - uvStep * KERNEL_RADIUS;
    for (int it = 0; it <= KERNEL_RADIUS * 2; ++it)
    {
        // calculate weights
        //float spatialWeight = 1.0;
        float spatialWeight = kernel(KERNEL_RADIUS - it, SPATIAL_VARIANCE);
        float2 sampleAO = getColor(uv).xy;
        float2 sampleAOSq = sampleAO * sampleAO;
        if (dir.y != 0) // second iteration
            sampleAOSq = getColor(uv).zw; // squared values will be in the last two channels
        
        //float importanceWeight = pow(max(BRIGHT(sampleAO) - DARK(sampleAO), 1e-8), IMPORTANCE_EXPONENT);
        //float importanceWeight = pow(saturate(1.0 - (BRIGHT(sampleAO) - DARK(sampleAO))), IMPORTANCE_EXPONENT);
        float importanceWeight = 1.0;
        //float relativeDepth = saturate(abs(getDepth(uv) / localDepth - 1.0)); // 0.0 = same depth
        float relativeDepth = abs(getDepth(uv) / localDepth - 1.0); // 0.0 = same depth
        float depthWeight = kernel(relativeDepth, DEPTH_VARIANCE);
        //float depthWeight = kernel(relativeDepth, 0.1);
        //float depthWeight = 1.0;

        // update sum
        float w = spatialWeight * depthWeight; // * importanceWeight
        weightSum += w;
        means += w * sampleAO;
        meansSq += w * sampleAOSq;
        
        uv += uvStep;
    }
    
    means /= max(weightSum, 1e-4);
    meansSq /= max(weightSum, 1e-4);
    if(weightSum < 1e-4)
    {        
        means = local; // fallback color when weight is too small
        meansSq = local * local;
    }
    
    //if (CLAMP_RESULTS) pixelSum = clamp(pixelSum, getDark(pIn.texC), getBright(pIn.texC));

    if (dir.y != 0) // last iteration
    {

        
        if (OUTPUT == OUTPUT_BRIGHT_MEAN)
            return means.x;
        if (OUTPUT == OUTPUT_DARK_MEAN)
            return means.y;
        
    // compute final color value from bright and dark image
        float c = BRIGHT(means.x); // use bright image as default
        float2 cOriginal = getOriginalColor(pIn.texC);
        if (OUTPUT == OUTPUT_BRIGHT)
            return cOriginal.x;
        if (OUTPUT == OUTPUT_DARK)
            return cOriginal.y;
        
        // local deviation
        float2 ldev = pow(abs(cOriginal - means), DEV_EXPONENT);
        ldev.y = max(ldev.y, DARK_EPSILON); // prevent division by zero
        ldev.x *= ENHANCE_CONTRAST; // contrast enhancement
        float2 w = ldev.yx / (ldev.x + ldev.y);
        if (LOCAL_DEVIATION)
            c = dot(cOriginal, w); // interpolate from original color value

        if (OUTPUT == OUTPUT_BRIGHT_LOCAL_DEV)
            return ldev.x;
        if (OUTPUT == OUTPUT_DARK_LOCAL_DEV)
            return ldev.y;

        // actual deviation
        float2 adev = pow(abs(meansSq - means * means), DEV_EXPONENT * 0.5);
        adev.y = max(adev.y, DARK_EPSILON); // prevent division by zero
        adev.x *= ENHANCE_CONTRAST; // contrast enhancement
        w = adev.yx / (adev.x + adev.y);
        if (!LOCAL_DEVIATION)
            c = dot(cOriginal, w); // interpolate from original color value
        
        if (OUTPUT == OUTPUT_BRIGHT_DEV)
            return adev.x;
        if (OUTPUT == OUTPUT_DARK_DEV)
            return adev.y;
        
    // clamp between bright and dark image
        //if (CLAMP_RESULTS)
        //    c = clamp(c, DARK(cOriginal), BRIGHT(cOriginal));

        return c;
    }
    
    return float4(means, meansSq);
}
