import Scene.Camera.Camera;

Texture2D<float2> gMotionVec;
Texture2D<float> gDepth;
Texture2D<float> gPrevDepth;
Texture2D<float> gAO;
Texture2D<float> gPrevAO;
Texture2D<uint> gPrevHistory;

Texture2D<uint> gStableMask;

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

int2 UVToPixel(float2 uv)
{
    float width, height;
    gDepth.GetDimensions(width, height);
    return int2(floor(uv * float2(width, height)));
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

struct PsOut
{
    float ao : SV_Target0;
    uint n : SV_Target1; // history
};

PsOut main(float2 texC : TEXCOORD, float4 svPos : SV_POSITION)
{
    float depth = gDepth.SampleLevel(gDepthSampler, texC, 0.0);
    PsOut o;
    o.ao = gAO.SampleLevel(gAOSampler, texC, 0.0);
    o.n = 1;
    //float3 posV = UVToViewSpace(texC, depth);

    float2 mvec = gMotionVec.SampleLevel(gDepthSampler, texC, 0.0);
    
    if (!isInValidArea(texC + mvec)) // test if motion vector is outside of the screen
    {
        return o; // reset history, use new value
    }

    int2 prevPixel = UVToPixel(texC + mvec);
    //float prevRawDepth = gPrevDepth.SampleLevel(gDepthSampler, texC + mvec, 0.0);
    float prevRawDepth = gPrevDepth[prevPixel];
    float3 prevPosV = mul(prevViewToCurView, float4(UVToViewSpace(texC + mvec, prevRawDepth), 1.0)).xyz;
    float prevDepth = -prevPosV.z;

    bool isStablePixel = gStableMask[int2(svPos.xy)] != 0;
    if (RelativeDepth(depth, prevDepth) < 0.1 && !isStablePixel)
    {
        float prevAO = gPrevAO.SampleLevel(gAOSampler, texC + mvec, 0.0);
        //uint prevN = gPrevAO.SampleLevel(gAOSampler, texC + mvec, 0.0);
        //float prevAO = gPrevAO[prevPixel];
        uint prevN = gPrevHistory[prevPixel];
        o.ao = (prevN * prevAO + o.ao) / (prevN + 1);
        o.n = min(prevN + 1u, 30u);
        //o.ao = o.n / 128.0;
    }
    //else o.ao = 0.0;
    
    return o;
}
