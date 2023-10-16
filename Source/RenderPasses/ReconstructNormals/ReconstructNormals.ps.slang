import Utils.Math.PackedFormats;
import Scene.Camera.Camera;

//Texture2D<float3> gNormals;
Texture2D<float> gDepth;
SamplerState gSampler;

cbuffer StaticCB
{
    Camera gCamera;
    float2 duv; // difference in uv for 1 pixel (1/resolution)
}

float3 UVToViewSpace(float2 uv, float viewDepth)
{
    float2 ndc = float2(uv.x, 1.0 - uv.y) * 2.0 - 1.0; // normalized device coordinates [-1, 1]
    const float2 imageScale = 0.5 * float2(gCamera.data.frameWidth / gCamera.data.focalLength, gCamera.data.frameHeight / gCamera.data.focalLength);
    return float3(ndc * viewDepth * imageScale, -viewDepth);
}

// Idea from HBAO+: take lowest gradient of depth to get gradient for normal
float3 MinDiff(float3 P, float3 Pr, float3 Pl)
{
    float3 grad1 = Pr - P;
    float3 grad2 = P - Pl;
    return (dot(grad1, grad1) < dot(grad2, grad2)) ? grad1 : grad2;
}

#if COMPRESS
#define return_t uint
#else
#define return_t float3
#endif

return_t main(float2 uv : TEXCOORD, float4 svPos : SV_POSITION) : SV_TARGET
{
    float3 P = UVToViewSpace(uv, gDepth.SampleLevel(gSampler, uv, 0.0));
    float2 uvr = uv + float2(duv.x, 0.0);
    float3 Pr = UVToViewSpace(uvr, gDepth.SampleLevel(gSampler, uvr, 0.0));
    float2 uvl = uv + float2(-duv.x, 0.0);
    float3 Pl = UVToViewSpace(uvl, gDepth.SampleLevel(gSampler, uvl, 0.0));
    float2 uvt = uv + float2(0.0, duv.y);
    float3 Pt = UVToViewSpace(uvt, gDepth.SampleLevel(gSampler, uvt, 0.0));
    float2 uvb = uv + float2(0.0, -duv.y);
    float3 Pb = UVToViewSpace(uvb, gDepth.SampleLevel(gSampler, uvb, 0.0));

    float3 normal = normalize(cross(MinDiff(P, Pr, Pl), MinDiff(P, Pt, Pb)));
    #if COMPRESS
    
    uint compressed = 0;
    #if USE_16_BIT
    compressed = encodeNormal2x8(normal);
    #else
    compressed = encodeNormal2x16(normal);
    #endif
    return compressed;

    #else // not compressed
    return normal;
    #endif
}
