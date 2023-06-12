#ifndef type
#define type float
#endif

Texture2DArray<type> src;

type main(float2 uv : TEXCOORD, float4 svPos : SV_POSITION) : SV_TARGET
{
    uint2 xy = uint2(svPos.xy); // full resolution pos
    uint2 xyQuarter = xy / 4u;
    uint2 offset = xy % 4u;
    uint slice = offset.y * 4u + offset.x;

    return src[uint3(xyQuarter, slice)];
}
