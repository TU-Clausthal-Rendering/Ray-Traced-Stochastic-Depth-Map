import Scene.Raster;

VSOut vsMain(VSIn vIn)
{
    return defaultVS(vIn);
}

//static const uint stratifiedIndices[] = { 0, 1, 17, 137, 697, 2517, 6885, 14893, 26333, 39203, 50643, 58651, 63019, 64839, 65399, 65519, 65535, };
//static const uint stratifiedIndices[] = { 0, 1, 9, 37, 93, 163, 219, 247, 255 };
StructuredBuffer<uint> stratifiedIndices;
StructuredBuffer<uint> stratifiedLookUpTable;
Texture2D<float> depthBuffer;

cbuffer CameraCB
{
    float zNear;
    float zFar;
};

// Hash function from "Improved Alpha Testing Using Hashed Sampling"
float hash(float2 v)
{
    return frac(1.0e4 * sin(17.0 * v.x + 0.1 * v.y) * (0.1 + abs(sin(13.0 * v.y + v.x))));
}

// 3D hash function from "Improved Alpha Testing Using Hashed Sampling"
float hash3D(float3 v)
{
    return hash(float2(hash(v.xy), v.z));
}

// 4D hash function based on the 3D hash function
float hash4D(float4 v)
{
    return hash(float2(hash3D(v.xyz), v.w));
}

struct PsOut
{
    uint SampleMask[1] : SV_Coverage; // let this be an array to avoid compiler bugs
#ifdef LINEARIZE
    float depth : SV_Depth;
#endif
};

[earlydepthstencil]
PsOut psMain(VSOut vsOut, uint triangleIndex : SV_PrimitiveID)
{
    if (alphaTest(vsOut, triangleIndex, ImplicitLodTextureSampler()))
        discard;
    PsOut output;

    // assume that a depth-map of the first layer is available => discard fragments with the same depth
    float firstDepth = depthBuffer[uint2(vsOut.posH.xy)];
    if (vsOut.posH.z <= firstDepth)
        discard;

    // #if NON_STOCHASTIC_FIRST_LAYER // Potentially, you could create the regular depth map in this same pass by never discarding the first sample
    // 	if (gl_SampleID == 0) { return; } 
    // #endif

    float rng = hash4D(float4(vsOut.posW.xyz, 1.438943289));

    int R = int(floor(ALPHA * NUM_SAMPLES + rng));

    if (R >= NUM_SAMPLES)
    {
        output.SampleMask[0] = 0xffff; // This code assumes maximum 32 samples, otherwise further gl_SampleMask values need to be set
    }
    else if (R != 0)
    {
        float rng2 = hash3D(vsOut.posW.zyx);
        int index = int(lerp(stratifiedIndices[R], stratifiedIndices[R + 1], rng2));
        uint mask = stratifiedLookUpTable[index];
        output.SampleMask[0] = int(mask);
    }
    else
        discard;

    #ifdef LINEARIZE
    output.depth = (vsOut.posH.w - zNear) / (zFar - zNear); // linear depth normalizedto [0,1] because a depth target cannot store higher values
    #endif

    return output;
}
