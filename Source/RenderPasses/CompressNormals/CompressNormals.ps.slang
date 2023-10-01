import Utils.Math.PackedFormats;

Texture2D<float3> gNormals;

cbuffer PerFrameCB
{
    float4x4 gViewMat;
}

uint main(float2 uv : TEXCOORD, float4 svPos : SV_POSITION) : SV_TARGET
{
    float3 normal = gNormals[uint2(svPos.xy)].xyz;
    #if VIEW_SPACE
    normal = mul(float3x3(gViewMat), normal);
    #endif

    uint compressed = 0;
    #if USE_16_BIT
    compressed = encodeNormal2x8(normal);
    #else
    compressed = encodeNormal2x16(normal);
    #endif
    return compressed;
}
