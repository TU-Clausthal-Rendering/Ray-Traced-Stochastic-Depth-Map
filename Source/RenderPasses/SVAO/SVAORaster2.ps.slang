import Scene.RaytracingInline;
#include "Common.slang"

// inputs from previous stage
Texture2D<uint> aoMask;
Texture2D<float> aoPrev;

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
            const uint materialID = gScene.getMaterialID(hit.instanceID);
            const MaterialHeader header = gScene.materials.materialData[materialID].header;

            bool isAlphaTested = header.getAlphaMode() == AlphaMode::Mask;

                    // needs alpha testing?
            if (isAlphaTested)
            {
                const VertexData v = gScene.getVertexData(hit);
                if (gScene.materials.alphaTest(v, materialID, 0.0)) // TODO correct lod?   
                    continue; // alpha test failed => ignore this triangle
            }

            frontFace = frontFace || isAlphaTested || header.isDoubleSided();
            if (!frontFace)
                continue; // this is just for rasterizer compability
                        
            if (t <= rayData.tSphereStart)
            {
                rayData.tLastFrontFaceHalo = max(rayData.tLastFrontFaceHalo, t);
                if (t >= rayData.tConstRadiusStart)
                    break; // we can stop the query, because this will set the visibility to zero
            }
            else // inside sphere
            {
                rayData.tFirstFrontFaceInside = min(rayData.tFirstFrontFaceInside, t);
                rayQuery.CommitNonOpaqueTriangleHit(); // since we save the min, we can commit here
            }
                        
        }
    }
}

[earlydepthstencil]
float main(float2 texC : TEXCOORD, float4 svPos : SV_POSITION) : SV_TARGET
{
    uint mask = aoMask[uint2(svPos.xy)] & 0xFF;
    
    float visibility = calcAO2(uint2(svPos.xy), mask);
    float prevAo = aoPrev.SampleLevel(gTextureSampler, texC, 0);
    visibility += prevAo;
    visibility = pow(visibility, gData.exponent);
    return visibility;
}
