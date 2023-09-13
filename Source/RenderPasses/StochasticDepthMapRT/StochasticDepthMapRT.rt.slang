import Scene.Raytracing;
import Rendering.Materials.TexLODHelpers;
import Rendering.Materials.TexLODTypes;

RWTexture2DArray<float> depthOutTex;
Texture2D<uint> maskTex;

// Note: using lods (ray cones) is about 20% faster than sampling lod 0 (no lod)
#define TEXTURE_LOD_T ExplicitRayConesLodTextureSampler
//#define TEXTURE_LOD_T ExplicitLodTextureSampler
#include "Common.slangh"

struct RayData // cannot be compressed to half floats => no diff in rendering time + insufficient visual quality
{
    float depths[NUM_SAMPLES];
    uint count; // number of processed samples
};

//ExplicitLodTextureSampler computeLod(VertexData v, float3 rayDir) { return ExplicitLodTextureSampler(0.0); }

ExplicitRayConesLodTextureSampler computeLod(VertexData v, float3 rayDir)
{
    RayCone rc = RayCone(0.0, RAY_CONE_SPREAD);
    rc = rc.propagateDistance(RayTCurrent());
    float lambda = rc.computeLOD(v.coneTexLODValue, rayDir, v.normalW);
    return ExplicitRayConesLodTextureSampler(lambda);
}

[shader("anyhit")]
void anyHit(inout RayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    bool commit = algorithm(rayData.depths, rayData.count, attribs.barycentrics, RayTCurrent(), getGeometryInstanceID(), PrimitiveIndex(), WorldRayDirection());
    if (!commit)
        IgnoreHit();
}

[shader("miss")]
void miss(inout RayData rayData)
{
}

[shader("closesthit")]
void closestHit(inout RayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    // is skipped
}

[shader("raygeneration")]
void rayGen()
{
    uint2 svPos = DispatchRaysIndex().xy;

    uint mask = maskTex[svPos] & 0xFF;
    if (mask == 0u)
        return;

    RayDesc ray = gScene.camera.computeRayPinhole(DispatchRaysIndex().xy, DispatchRaysDimensions().xy).toRayDesc();
    float epsilon = 0.1 * gScene.camera.data.nearZ; // small ray offset
    ray.TMin = ViewDepthToRay(ray.Direction, depthInTex[svPos]) + epsilon; // start after first known hit

    uint iRayMin = rayMinTex[svPos];
    if (iRayMin != 0u) // test if valid values
        ray.TMin = max(asfloat(iRayMin), ray.TMin);
    
    uint iRayMax = rayMaxTex[svPos];
    if(iRayMax != 0u) // test if valid values
        ray.TMax = min(asfloat(iRayMax), ray.TMax);
    
    RayData rayData;
    rayData.count = 0;
    for (uint i = 0; i < NUM_SAMPLES; ++i)
        rayData.depths[i] = DEFAULT_DEPTH;
    
    // TODO custom culling?
    TraceRay(
        gScene.rtAccel,
        RAY_FLAG_CULL_BACK_FACING_TRIANGLES | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_FORCE_NON_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
        0xff /* instanceInclusionMask */, 0 /* hitIdx */, 1 /*ray type count*/, 0 /* missIdx */,
        ray, rayData
    );

    for (uint i = 0; i < NUM_SAMPLES; ++i)
        depthOutTex[uint3(svPos, i)] = rayData.depths[i];
}
