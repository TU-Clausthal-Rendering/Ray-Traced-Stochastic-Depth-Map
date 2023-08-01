import Scene.Camera.Camera;

Texture2D<float> gDepth;
Texture2D<float> gPrevDepth; // prev depth OR prev depth 2

SamplerState gLinearSampler;
SamplerState gPointSampler;

cbuffer PerFrameCB
{
    Camera gCamera;
    float4x4 prevViewToCurView; // viewMat * Inverse(prevViewMat)
    float4x4 curViewToPrevView; // prevViewMat * Inverse(viewMat)
    uint2 resolution; // screen resolution
    float minSeparationDist;
}



int2 UVToPixel(float2 uv)
{
    return int2(floor(uv * float2(resolution)));
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

struct VSOut
{
    float2 texC : TEXCOORD;
    float originalDepth : DEPTH;
    float4 svPos : SV_POSITION;
};

VSOut vsMain(uint id : SV_VertexID)
{
    VSOut o;
    //uint2 pixel = uint2(id % resolution.x, id / resolution.x);
    uint2 pixel = uint2(id % resolution.x,  (id / resolution.x));
    o.texC = (pixel + 0.5) / float2(resolution);
    float depth = gPrevDepth[pixel];
    o.originalDepth = depth;
    float3 prevPosV = UVToViewSpace(o.texC, depth);
    // transform to current view space and then project it
    float3 curPosV = mul(prevViewToCurView, float4(prevPosV, 1.0)).xyz;

    // apply projection
    o.svPos = mul(gCamera.data.projMat, float4(curPosV, 1.0));
    float4 svPos = o.svPos;
    //o.svPos.z = 0.0;
    //o.svPos = float4(o.texC, 0.0, 1.0);

    //o.texC.r = 10.0; //svPos.y / svPos.w;
    return o;
}

struct PsOut
{
    float depth2 : SV_Target0;
};

PsOut psMain(float2 texC : TEXCOORD, float originalDepth : DEPTH, float4 svPos : SV_POSITION)
{
    //float depth = gDepth.SampleLevel(gLinearSampler, , 0.0);
    float primaryDepth = gDepth[int2(svPos.xy)];
    PsOut o;
    o.depth2 = svPos.w;

    // discard far plane points
    if(svPos.w > 0.99 * gCamera.data.farZ)
        discard;

    // depth peeling
    if (o.depth2 < primaryDepth + minSeparationDist)
        discard;

    return o;
}
