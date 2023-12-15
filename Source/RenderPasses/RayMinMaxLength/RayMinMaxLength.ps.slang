Texture2D<uint> gRayMin;
Texture2D<uint> gRayMax;

float main(float2 uv : TEXCOORD, float4 svPos : SV_POSITION) : SV_TARGET
{
    uint iRayMin = gRayMin[int2(svPos.xy)];
    uint iRayMax = gRayMax[int2(svPos.xy)];
    float rayMin = asfloat(iRayMin);
    float rayMax = asfloat(iRayMax);

    if (iRayMax == 0u)
        return 0.0f;

    return max(0.0, rayMax - rayMin) / 32.0;
}
