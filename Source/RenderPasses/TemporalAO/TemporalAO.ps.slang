import Scene.Camera.Camera;

Texture2D<float2> gMotionVec;
Texture2D<float> gDepth;
Texture2D<float> gPrevDepth;
Texture2D<float> gAO;
Texture2D<float> gPrevAO;

SamplerState gDepthSampler;
SamplerState gAOSampler;

cbuffer PerFrameCB
{
    Camera gCamera;
    float4x4 prevViewToCurView; // viewMat * Inverse(prevViewMat)
    float2 uvMin; // uv coordinate clamping when using a guard band
    float2 uvMax;
}

bool isInValidArea(float2 uv)
{
    return uv.x >= uvMin.x && uv.x <= uvMax.x && uv.y >= uvMin.y && uv.y <= uvMax.y;
}

// uv: uv coordinates [0, 1]
// viewDepth: linear depth in view space (positive z)
// return: view space position (negative z)
float3 UVToViewSpace(float2 uv, float viewDepth)
{
    float2 ndc = float2(uv.x, 1.0 - uv.y) * 2.0 - 1.0; // normalized device coordinates [-1, 1]
    const float2 imageScale = 0.5 * float2(gCamera.data.frameWidth / gCamera.data.focalLength, gCamera.data.frameHeight / gCamera.data.focalLength);
    return float3(ndc * viewDepth * imageScale, -viewDepth);
}

// posV: view space position (negative z)
// return: texture uv [0, 1]
float2 ViewSpaceToUV(float3 posV)
{
    const float2 imageScale = 0.5 * float2(gCamera.data.frameWidth / gCamera.data.focalLength, gCamera.data.frameHeight / gCamera.data.focalLength);
    float2 ndc = posV.xy / (imageScale * posV.z);
    return ndc * float2(-0.5, 0.5) + 0.5; // since posV.z is negative, the sign order is inversed
}

float RelativeDepth(float reference, float value)
{
    return abs(1.0 - value / reference);
}

float main(float2 texC : TEXCOORD) : SV_TARGET0
{
    float depth = gDepth.SampleLevel(gDepthSampler, texC, 0.0);
    float ao = gAO.SampleLevel(gAOSampler, texC, 0.0);
    //float3 posV = UVToViewSpace(texC, depth);

    float2 mvec = gMotionVec.SampleLevel(gDepthSampler, texC, 0.0);
    
    if (!isInValidArea(texC + mvec)) // test if motion vector is outside of the screen
    {
        return ao; // reset history, use new value
    }
    
    float prevRawDepth = gPrevDepth.SampleLevel(gDepthSampler, texC + mvec, 0.0);
    float3 prevPosV = mul(prevViewToCurView, float4(UVToViewSpace(texC + mvec, prevRawDepth), 1.0)).xyz;
    float prevDepth = -prevPosV.z;

    bool isStablePixel = false;
    if (RelativeDepth(depth, prevDepth) < 0.1 && !isStablePixel)
    {
        float prevAO = gPrevAO.SampleLevel(gAOSampler, texC + mvec, 0.0);
        return lerp(ao, prevAO, 0.9);
    }
    
    return ao;
}
