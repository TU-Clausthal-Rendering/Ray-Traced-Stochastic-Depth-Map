import Scene.Material.MaterialData;

Texture2D<uint4> gMtlData;

uint main(float2 uv : TEXCOORD, float4 svPos : SV_POSITION) : SV_TARGET
{
    uint4 mtlData = gMtlData[int2(svPos.xy)];
    MaterialHeader matHeader;
    matHeader.packedData = mtlData.yz;

    return matHeader.isDoubleSided() ? 1 : 0;
}
