import Scene.RaytracingInline;
#include "Common.slang"

// inputs from previous stage
Texture2D<uint> aoMask;
RWTexture2D<unorm ao_t> aoPrev;

// implementation of the traceAORay used inside calcAO2
void traceAORay(RayDesc ray, inout RayData rayData)
{
    // skip procedural and force all triangle to be handled by any-hit traversal
    #if AO_KERNEL == AO_KERNEL_VAO
    RayQuery < RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_FORCE_NON_OPAQUE > rayQuery;
    #else // HBAO
    RayQuery < RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES > rayQuery;
    #endif
    rayQuery.TraceRayInline(gScene.rtAccel, RAY_FLAG_NONE, 0xff, ray);

    while (rayQuery.Proceed())
    {
        if (rayQuery.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            // extract hit properties
            float t = rayQuery.CandidateTriangleRayT();
            
            #if AO_KERNEL == AO_KERNEL_VAO
            if (t < rayData.tLastFrontFaceHalo)
                continue; // we can skip this since it would not contribute anyways (unfortunately there is no option to set the ray.min afterwards)
            #endif

            bool frontFace = rayQuery.CandidateTriangleFrontFace();
            const TriangleHit hit = getCandidateTriangleHit(rayQuery);

            uint res = aoAnyHit(rayData, t, hit, frontFace, ray.Direction);
            if (res == AO_HIT_ACCEPT)
                rayQuery.CommitNonOpaqueTriangleHit();
            if (res == AO_HIT_ACCEPT_AND_END)
                break;      
        }
    }

    #if AO_KERNEL == AO_KERNEL_HBAO
    if(rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        rayData.tFirst = rayQuery.CommittedRayT();
    #endif
}

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID, uint3 localId : SV_GroupThreadID)
{
    uint2 offset = (group.xy / 2u) * 32u + 2u * localId.xy + (group.xy % 2u);
    //uint2 svPos = offset + uint2(guardBand); // pixel position
    uint2 svPos = id.xy + uint2(guardBand); // pixel position
    float2 texC = (svPos + float2(0.5)) * gData.invResolution; // texture position
    
    uint mask = aoMask[uint2(svPos.xy)] & 0xFF;
    if(mask == 0)
        return;
    
    ao_t visibility = calcAO2(svPos, mask);
    visibility += float(aoPrev[svPos]);
    darkmap(visibility.y = min(visibility.x, visibility.y)); // make sure that bright ao is bigger than dark ao
    visibility = BasicAOData.finalize(visibility);
    aoPrev[svPos] = visibility;
}
