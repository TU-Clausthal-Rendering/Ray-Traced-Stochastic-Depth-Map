import Scene.RaytracingInline;
#include "Common.slang"

// inputs from previous stage
Texture2D<uint> aoMask;
Texture2D<ao_t> aoPrev;

// implementation of the traceAORay used inside calcAO2
void traceAORay(RayDesc ray, inout RayData rayData)
{
               // skip procedural and force all triangle to be handled by any-hit traversal
    RayQuery < RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_FORCE_NON_OPAQUE > rayQuery;
    rayQuery.TraceRayInline(gScene.rtAccel, RAY_FLAG_NONE, 0xff, ray);

    while (rayQuery.Proceed())
    {
        if (rayQuery.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            // extract hit properties
            float t = rayQuery.CandidateTriangleRayT();
            if (t < rayData.tLastFrontFaceHalo)
                continue; // we can skip this since it would not contribute anyways (unfortunately there is no option to set the ray.min afterwards)

            bool frontFace = rayQuery.CandidateTriangleFrontFace();
            const TriangleHit hit = getCandidateTriangleHit(rayQuery);

            uint res = aoAnyHit(rayData, t, hit, frontFace);
            if (res == AO_HIT_ACCEPT)
                rayQuery.CommitNonOpaqueTriangleHit();
            if (res == AO_HIT_ACCEPT_AND_END)
                break;      
        }
    }
}

[earlydepthstencil]
ao_t main(float2 texC : TEXCOORD, float4 svPos : SV_POSITION) : SV_TARGET
{
    uint mask = aoMask[uint2(svPos.xy)] & 0xFF;
    
    ao_t visibility = calcAO2(uint2(svPos.xy), mask);
    ao_t prevAo = aoPrev.SampleLevel(gTextureSampler, texC, 0);
    visibility += prevAo;
    darkmap(visibility.y = min(visibility.x, visibility.y)); // make sure that bright ao is bigger than dark ao
    visibility = pow(visibility, gData.exponent);
    return visibility;
}
