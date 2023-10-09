import Scene.RaytracingInline;
import Scene.Raster;
import Rendering.Lights.LightHelpers;

Texture2D<float4> gPos; // indicates depth accesses for the secondary pass

//Static Arrays for Poission Disc Sampling
static const float2 kPoissonDisc[16] =
{
    float2(-0.94201624, -0.39906216), float2(0.94558609, -0.76890725),
     float2(-0.094184101, -0.92938870), float2(0.34495938, 0.29387760),
     float2(-0.91588581, 0.45771432), float2(-0.81544232, -0.87912464),
     float2(-0.38277543, 0.27676845), float2(0.97484398, 0.75648379),
     float2(0.44323325, -0.97511554), float2(0.53742981, -0.47373420),
     float2(-0.26496911, -0.41893023), float2(0.79197514, 0.19090188),
     float2(-0.24188840, 0.99706507), float2(-0.81409955, 0.91437590),
     float2(0.19984126, 0.78641367), float2(0.14383161, -0.14100790)
};

float calcVisibility(RayDesc ray)
{
    float visibility = 1.0;
    
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
            //if (gScene.materials.alphaTest(v, materialID, 0.0)) // TODO correct lod?   
            //    continue; // alpha test failed => ignore this triangle
            //bool alphaTest = gScene.materials.materialData[materialID].header.getAlphaMode() == AlphaMode::Mask;
            //if (!alphaTest)
            //    return 0.0;
            
            let material = gScene.materials.getMaterial(materialID);
            let lod = ExplicitLodTextureSampler(6.0);
            float opacity = material.evalOpacity(gScene.materials, v, lod); // TODO lod?

            visibility = visibility * (1.0 - opacity);
            if (visibility < 1e-8)
                break;
            //rayQuery.CommitNonOpaqueTriangleHit();
        }
        else
            return 0.0; // hit opaque => occluded
    }

    if (rayQuery.CommittedStatus() != COMMITTED_NOTHING)
        visibility = 0.0;
    
    return visibility;
}

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

    // send initial ray
    float visibility = calcVisibility(ray);
    return visibility;

    float4 vquad;
    vquad.x = QuadReadLaneAt(visibility, 0);
    vquad.y = QuadReadLaneAt(visibility, 1);
    vquad.z = QuadReadLaneAt(visibility, 2);
    vquad.w = QuadReadLaneAt(visibility, 3);
    
    if (vquad.x == vquad.y && vquad.y == vquad.z && vquad.z == vquad.w) // all the same?
        return visibility;
    
    float3 up = float3(0.0, 1.0, 0.0);
    float3 normal = ray.Direction;
    float3 tangent = normalize(cross(up, ray.Direction));
    float3 bitangent = cross(tangent, ray.Direction);

    //float visibility = calcVisibility(ray);
    float wsum = 1.0f;

    for (int i = 0; i < 16; ++i)
    {
        float w = 1.0;
        float strength = 0.001;
        float3 newDir = normal + (kPoissonDisc[i].x * tangent + kPoissonDisc[i].y * bitangent) * strength;
        ray.Direction = normalize(newDir);
        visibility += w * calcVisibility(ray);
        wsum += w;
    }

    visibility = visibility / wsum;
    
    //visibility = (QuadReadAcrossX(visibility) + QuadReadAcrossY(visibility) + visibility * 2.0) / 4.0;
    
    return visibility;
}
