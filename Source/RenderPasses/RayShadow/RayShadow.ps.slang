import Scene.RaytracingInline;
import Scene.Raster;
import Rendering.Lights.LightHelpers;
import Rendering.Materials.TexLODTypes;
import Rendering.Materials.TexLODHelpers;

Texture2D<float4> gPos; // indicates depth accesses for the secondary pass
Texture2D<float4> gNormal;

float calcVisibility(RayDesc ray, RayCone rc, float2 frameDim)
{
    float visibility = 1.0;

    float visibilityCutoff = 0.01;
    
    // TODO RAY_FLAG_CULL_BACK_FACING_TRIANGLES
    RayQuery < RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH > rayQuery;
    rayQuery.TraceRayInline(gScene.rtAccel, RAY_FLAG_NONE, 0xff, ray);
    while (rayQuery.Proceed())
    {
        if (rayQuery.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            float t = rayQuery.CandidateTriangleRayT();
            // alpha test
            const TriangleHit hit = getCandidateTriangleHit(rayQuery);
            const uint materialID = gScene.getMaterialID(hit.instanceID);
            const VertexData v = gScene.getVertexData(hit);
           
            let material = gScene.materials.getMaterial(materialID);

            // LOD
            RayCone newRc = rc.propagateDistance(t);
            newRc = newRc.hit(ray.Direction, v.faceNormalW);
            float lambda = newRc.computeLOD(v.coneTexLODValue);
            //lambda += 1.0; // small offset to make shadows a bit more soft
            let lod = ExplicitRayConesLodTextureSampler(lambda);
            //let lod = ExplicitLodTextureSamplers(0.0);
            float opacity = material.evalOpacity(gScene.materials, v, lod);
            opacity = saturate(opacity * 2.0); // scale opacity because opacity > 0.5 is opaque for alpha test
            visibility = visibility * (1.0 - opacity);
            if (visibility < visibilityCutoff)
                return 0.0;
        }
        else
            return 0.0; // hit opaque => occluded
    }

    if (rayQuery.CommittedStatus() != COMMITTED_NOTHING)
        visibility = 0.0;

    // scale based on the cutoff
    visibility = saturate((visibility - visibilityCutoff) / (1.0 - visibilityCutoff));
    
    return visibility;
}

float main(float2 texC : TEXCOORD, float4 svPos : SV_POSITION) : SV_Target
{
    float4 posW = gPos[int2(svPos.xy)];
    if (posW.w == 0.0)
        return 1.0; // background
    
    if (gScene.getLightCount() == 0)
        return 1.0;

    float3 faceNormalW = gNormal[int2(svPos.xy)].xyz;

    float2 frameDim;
    gPos.GetDimensions(frameDim.x, frameDim.y);
    
    LightData light = gScene.getLight(0);
    RayDesc ray;
    ray.Origin = posW.xyz;
    ray.TMin = gScene.camera.data.nearZ;

    RayCone rc = RayCone(0.0, RAY_CONE_SPREAD);
    float3 V = gScene.camera.getPosition() - posW.xyz;
    float lenV = length(V);
    rc = rc.propagateDistance(lenV); // travel t
    rc = rc.hit(V/lenV, faceNormalW); // hit surface
    
    
    float secondarySpreadAngle = RAY_CONE_SPREAD;
    if (light.type == uint(LightType::Point))
    {
        ray.Direction = normalize(light.posW - posW.xyz);
        ray.TMax = distance(light.posW, posW.xyz);

        rc = rc.reflect(ray.Direction, faceNormalW);
        rc = rc.focus(distance(light.posW, posW.xyz));
    }
    else if (light.type == uint(LightType::Directional))
    {
        ray.Direction = -light.dirW;
        ray.TMax = gScene.camera.data.farZ;
        secondarySpreadAngle = 0.0; // light rays are orthogonal

        rc = rc.reflect(ray.Direction, faceNormalW);
        rc = rc.orthogonalize();
    }
    
    // send initial ray
    float visibility = calcVisibility(ray, rc, frameDim);
    //visibility = (QuadReadAcrossX(visibility) + QuadReadAcrossY(visibility) + visibility * 2.0) / 4.0;
    return visibility;
}
