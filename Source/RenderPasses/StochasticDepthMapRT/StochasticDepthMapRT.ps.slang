import Scene.RaytracingInline;
import Scene.Raster;

Texture2D<float> depthInTex;

StructuredBuffer<uint> stratifiedIndices;
StructuredBuffer<uint> stratifiedLookUpTable;

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

float RayToViewDepth(float3 rayDir, float rayT)
{
    float cosTheta = dot(normalize(gScene.camera.data.cameraW), rayDir);
    return rayT * cosTheta;
}

float ViewDepthToRay(float3 rayDir, float viewDepth)
{
    float cosTheta = dot(normalize(gScene.camera.data.cameraW), rayDir);
    return viewDepth / cosTheta;    
}

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

[earlydepthstencil]
PsOut main(float2 texC : TEXCOORD, float4 svPos : SV_POSITION)
{    
    float depths[NUM_SAMPLES];
    for (uint i = 0; i < NUM_SAMPLES; ++i)
        depths[i] = NORMALIZE ? 1.0 : 3.40282347E+37F;

    int width, height;
    depthInTex.GetDimensions(width, height);
    RayDesc ray = gScene.camera.computeRayPinhole(int2(svPos.xy), int2(width, height)).toRayDesc();
    float epsilon = 0.01 * gScene.camera.data.nearZ; // small ray offset
    ray.TMin = ViewDepthToRay(ray.Direction, depthInTex[int2(svPos.xy)]) + epsilon; // start after first known hit

    RayQuery < RAY_FLAG_CULL_BACK_FACING_TRIANGLES | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_FORCE_NON_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER > rayQuery;
    rayQuery.TraceRayInline(gScene.rtAccel, RAY_FLAG_NONE, 0xff, ray);
    while (rayQuery.Proceed())
    {
        if (rayQuery.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            // alpha test
            const TriangleHit hit = getCandidateTriangleHit(rayQuery);
            const uint materialID = gScene.getMaterialID(hit.instanceID);
            const VertexData v = gScene.getVertexData(hit);
            const MaterialHeader header = gScene.materials.materialData[materialID].header;

            // needs alpha testing?
            if (header.getAlphaMode() == AlphaMode::Mask)
            {
                const VertexData v = gScene.getVertexData(hit);
                if (gScene.materials.alphaTest(v, materialID, 0.0)) // TODO correct lod?   
                    continue; // alpha test failed => ignore this triangle
            }

            float t = rayQuery.CandidateTriangleRayT();
            t = RayToViewDepth(ray.Direction, t); // convert to view depth
    
            //float rng = hash4D(float4(DispatchRaysIndex().x, DispatchRaysIndex().y, t, 1.438943289));
            float rng = hash4D(float4(rayQuery.CandidateTriangleBarycentrics(), t, 1.438943289));

            int R = int(floor(ALPHA * NUM_SAMPLES + rng));
            uint coverageMask = 0;
    
            if (R >= NUM_SAMPLES)
            {
                coverageMask = 0xffff; // maximum 32 samples
            }
            else if (R != 0)
            {
                float rng2 = hash3D(float3(rayQuery.CandidateTriangleBarycentrics(), t));
                int index = int(lerp(stratifiedIndices[R], stratifiedIndices[R + 1], rng2));
                coverageMask = stratifiedLookUpTable[index];
            }

            if (NORMALIZE)
            {
                t = saturate((t - gScene.camera.data.nearZ) / (gScene.camera.data.farZ - gScene.camera.data.nearZ));
            }

            float maxT = 0.0;
            [unroll]
            for (uint i = 0; i < NUM_SAMPLES; ++i)
            {
                if (coverageMask & (1 << i))
                    depths[i] = min(depths[i], t); // z-test enabled
                maxT = max(maxT, depths[i]);
            }

            if (t >= maxT)
                rayQuery.CommitNonOpaqueTriangleHit();
        }
    }

    return PsOut(depths);
}
