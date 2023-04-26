Texture2D<uint> aoMask;

void main(float2 texC : TEXCOORD, float4 svPos : SV_POSITION)
{
    uint mask = aoMask[uint2(svPos.xy)];

    if (mask == 0u)
        discard;
}
