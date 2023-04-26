Texture2D<uint> mask;

void main(float2 uv : TEXCOORD, float4 svPos : SV_POSITION)
{
    uint2 xy = uint2(svPos.xy);

    if (mask[xy] == 0u)
        discard;
}
