import Scene.RaytracingInline;
import Scene.Raster;

#define TEXTURE_LOD_T ImplicitLodTextureSampler
#include "Common.slangh"

struct PsOut
{
    __init(float dephts[NUM_SAMPLES])
    {
        depths0 = dephts[0];
        #if NUM_SAMPLES > 1
        depths1 = dephts[1];
        #endif
        #if NUM_SAMPLES > 2
        depths2 = dephts[2];
        depths3 = dephts[3];
        #endif
        #if NUM_SAMPLES > 4
        depths4 = dephts[4];
        depths5 = dephts[5];
        depths6 = dephts[6];
        depths7 = dephts[7];
        #endif
    }
    
    float depths0 : SV_Target0;
#if NUM_SAMPLES > 1
    float depths1 : SV_Target1;
#endif
#if NUM_SAMPLES > 2
    float depths2 : SV_Target2;
    float depths3 : SV_Target3;
#endif
#if NUM_SAMPLES > 4
    float depths4 : SV_Target4;
    float depths5 : SV_Target5;
    float depths6 : SV_Target6;
    float depths7 : SV_Target7;
#endif
};

ImplicitLodTextureSampler computeLod(VertexData v, float3 rayDir)
{
    return ImplicitLodTextureSampler();
}

[earlydepthstencil]
PsOut main(float2 texC : TEXCOORD, float4 svPos : SV_POSITION)
{    
    float depths[NUM_SAMPLES];
    float3 normals[NUM_SAMPLES];
    uint count = 0;
    for (uint i = 0; i < NUM_SAMPLES; ++i)
        depths[i] = DEFAULT_DEPTH;

    int width, height;
    rayMinTex.GetDimensions(width, height);
    RayDesc ray = initRayDesc(uint2(svPos.xy), uint2(width, height));

    RayQuery < RAY_FLAG_CULL_BACK_FACING_TRIANGLES | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_FORCE_NON_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER > rayQuery;
    rayQuery.TraceRayInline(gScene.rtAccel, RAY_FLAG_NONE, 0xff, ray);
    while (rayQuery.Proceed())
    {
        if (rayQuery.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            bool commit = algorithm(depths, normals, count,
                rayQuery.CandidateTriangleBarycentrics(),
                rayQuery.CandidateTriangleRayT(),
                GeometryInstanceID(rayQuery.CandidateInstanceID(), rayQuery.CandidateGeometryIndex()),
                rayQuery.CandidatePrimitiveIndex(),
                ray.Direction
            );

            if(commit) rayQuery.CommitNonOpaqueTriangleHit();
        }
    }

    return PsOut(depths);
}
