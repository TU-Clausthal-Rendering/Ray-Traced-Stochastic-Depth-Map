import Scene.RaytracingInline;
import Scene.Raster;
import Rendering.Lights.LightHelpers;

Texture2D<float4> gPos; // indicates depth accesses for the secondary pass

float main(float2 texC : TEXCOORD, float4 svPos : SV_POSITION) : SV_Target
{
    float4 posW = gPos[int2(svPos.xy)];
    if (posW.w == 0.0)
        return 1.0; // background
    
    if (gScene.getLightCount() == 0)
        return 1.0;
    
    LightData light = gScene.getLight(0);
    RayDesc ray;
    ray.Origin = posW.xyz;
    ray.TMin = gScene.camera.data.nearZ;
    
    if (light.type == uint(LightType::Point))
    {
        ray.Direction = normalize(light.posW - posW.xyz);
        ray.TMax = distance(light.posW, posW.xyz);
    }
    else if (light.type == uint(LightType::Directional))
    {
        ray.Direction = -light.dirW;
        ray.TMax = gScene.camera.data.farZ;
    }

    // TODO RAY_FLAG_CULL_BACK_FACING_TRIANGLES
    RayQuery < RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH > rayQuery;
    rayQuery.TraceRayInline(gScene.rtAccel, RAY_FLAG_NONE, 0xff, ray);
    while (rayQuery.Proceed())
    {
        if (rayQuery.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            // alpha test
            const TriangleHit hit = getCandidateTriangleHit(rayQuery);
            const uint materialID = gScene.getMaterialID(hit.instanceID);
            const VertexData v = gScene.getVertexData(hit);
            if (gScene.materials.alphaTest(v, materialID, 0.0)) // TODO correct lod?   
                continue; // alpha test failed => ignore this triangle

            rayQuery.CommitNonOpaqueTriangleHit();
        }
    }

    float visibility = 1.0;
    if (rayQuery.CommittedStatus() != COMMITTED_NOTHING)
        visibility = 0.0;

    visibility = (QuadReadAcrossX(visibility) + QuadReadAcrossY(visibility) + visibility * 2.0) / 4.0;
    
    return visibility;
}
