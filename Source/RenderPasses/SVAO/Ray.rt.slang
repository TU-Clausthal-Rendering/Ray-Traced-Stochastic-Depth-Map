import Scene.Raytracing;
#include "Common.slang"

// inputs from previous stage
Texture2D<uint> aoMask;
RWTexture2D<ao_t> output;

[shader("anyhit")]
void anyHit(inout RayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    // extract hit properties
    float t = RayTCurrent();
    #if AO_KERNEL == AO_KERNEL_VAO
    if (t < rayData.tLastFrontFaceHalo)
        IgnoreHit(); // we can skip this since it would not contribute anyways (unfortunately there is no option to set the ray.min afterwards)
    #endif

    bool frontFace = HitKind() == HIT_KIND_TRIANGLE_FRONT_FACE;
    TriangleHit hit;
    hit.instanceID = getGeometryInstanceID();
    hit.primitiveIndex = PrimitiveIndex();
    hit.barycentrics = attribs.barycentrics;
    
    uint res = aoAnyHit(rayData, t, hit, frontFace, WorldRayDirection());
    if(res == AO_HIT_IGNORE)
        IgnoreHit();
    if (res == AO_HIT_ACCEPT_AND_END)
        AcceptHitAndEndSearch();

}

[shader("miss")]
void miss(inout RayData rayData)
{
}

[shader("closesthit")]
void closestHit(inout RayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    #if AO_KERNEL == AO_KERNEL_HBAO
    rayData.tFirst = RayTCurrent();
    #endif
}

// implementation of the traceAORay used inside calcAO2
void traceAORay(RayDesc ray, inout RayData rayData)
{
    TraceRay(
        gScene.rtAccel,
        #if AO_KERNEL == AO_KERNEL_VAO
        CULL_MODE_RAY_FLAG | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_FORCE_NON_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
        #else // HBAO
        CULL_MODE_RAY_FLAG | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES,
        #endif
        0xff /* instanceInclusionMask */, 0 /* hitIdx */, 1 /*ray type count*/, 0 /* missIdx */,
        ray, rayData
    );
}

[shader("raygeneration")]
void rayGen()
{
    uint2 svPos = DispatchRaysIndex().xy + uint2(guardBand);
    float2 texC = (float2(svPos) + 0.5) * gData.invResolution;

    uint mask = aoMask[svPos] & 0xFF;
    if (mask == 0u)
        return;

    ao_t visibility = calcAO2(svPos, mask);
    visibility += output[svPos];
    darkmap(visibility.y = min(visibility.x, visibility.y)); // make sure that bright ao is bigger than dark ao
    visibility = BasicAOData.finalize(visibility);
    output[svPos] = visibility;
}
