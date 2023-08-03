import Scene.Raster;

VSOut vsMain(VSIn vIn)
{
    return defaultVS(vIn);
}


Texture2D<float> prevDepth; // this is the linear depth

cbuffer CBuffer
{
    float minSeparationDistance;
};

void psMain(VSOut vsOut, uint triangleIndex : SV_PrimitiveID) : SV_TARGET
{
    if (alphaTest(vsOut, triangleIndex, ImplicitLodTextureSampler()))
        discard;

    float depth = vsOut.posH.w; // compare with linear depth
    const int2 uv = int2(vsOut.posH.xy);
    float prev = prevDepth[uv];

    // reject when smaller than previous depth
    if (depth <= prev + minSeparationDistance)
        discard;
}
