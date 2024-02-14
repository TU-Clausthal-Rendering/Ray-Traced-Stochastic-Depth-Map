import Scene.RaytracingInline;
import Scene.Raster;
import Rendering.Lights.LightHelpers;
import Rendering.Materials.TexLODTypes;
import Rendering.Materials.TexLODHelpers;

// gbuffer textures
Texture2D<float4> gPos;
Texture2D<float4> gNormal;

// active light index
cbuffer PerLightBuffer
{
    uint gLightIndex;
    float gPointLightClip;
    float gLodBias;
    bool gDiminishBorder;
};

SamplerState gSoftShadowSampler; // modified clamping mode to get softer borders

#ifndef USE_RAYCONES
#define USE_RAYCONES 1
#endif

#define RAY_CONE_SHADOW_RAY 0
#define RAY_CONE_SHADOW_ALPHA_TEST 1
#define RAY_CONE_SHADOW_SATURATED 2

#ifndef RAY_CONE_SHADOW
#define RAY_CONE_SHADOW RAY_CONE_SHADOW_SATURATED
#endif

struct SoftExplicitRayConesLodTextureSampler : ITextureSampler
{
    float rayconesLODWithoutTexDims;    ///< this is \Delta_t, which is texture independent, plus the rest of the terms, except the texture size, which is added below

    __init(float rayconesLODWithoutTexDims)
    {
        this.rayconesLODWithoutTexDims = rayconesLODWithoutTexDims;
    }

    float4 sampleTexture(Texture2D t, SamplerState s, float2 uv)
    {
        uint txw, txh;
        t.GetDimensions(txw, txh);
        float lambda = 0.5 * log2(txw * txh) + rayconesLODWithoutTexDims;

        float4 color = t.SampleLevel(gSoftShadowSampler, uv, lambda);
        if(!gDiminishBorder) return color;

        float lambdaBias = 1.0; // simulate higher mipmap to use border clamping earlier
        float w = float(txw) * pow(0.5, max(lambda + lambdaBias, 0.0));
        float h = float(txh) * pow(0.5, max(lambda + lambdaBias, 0.0));
        w = max(4.0, w); // make sure it does not diminish completely when the texture gets too small
        h = max(4.0, h);

        float2 pixel = uv * float2(w, h) + 0.5;
        if(all(pixel >= 1.0) && all(pixel <= float2(w, h)))
            return color;

        // border handling
        float2 t2D = max(1.0 - pixel, 0.0) + max(pixel - float2(w, h), 0.0);
        float tSum = saturate(t2D.x + t2D.y);
        color = max(lerp(color, -0.5, tSum), 0.0);
        return color;
    }
};

float traceShadowRay(
RayDesc ray
#if USE_RAYCONES
, RayCone rc
#endif
)
{
    float visibility = 1.0;
    float visibilityCutoff = 0.01; // visibility below this threshold is considered to be zero
    
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
            #if USE_RAYCONES
            RayCone shadowCone = rc.propagateDistance(t);
            shadowCone = shadowCone.hit(ray.Direction, v.faceNormalW);
            float lambda = shadowCone.computeLOD(v.coneTexLODValue);
            lambda += gLodBias;
            //lambda += 1.0; // small offset to make shadows a bit more soft
            let lod = SoftExplicitRayConesLodTextureSampler(lambda);
            float opacity = material.evalOpacity(gScene.materials, v, lod);
            
            if (RAY_CONE_SHADOW == RAY_CONE_SHADOW_SATURATED)
                opacity = saturate(opacity * 2.0); // scale opacity because opacity > 0.5 is opaque for alpha test
            else if(RAY_CONE_SHADOW == RAY_CONE_SHADOW_ALPHA_TEST)
                opacity = opacity < 0.5 ? 0.0 : 1.0;
            
            visibility = visibility * (1.0 - opacity);
            if (visibility < visibilityCutoff)
                return 0.0;
            
            #else
            let lod = ExplicitLodTextureSampler(gLodBias);
            float opacity = material.evalOpacity(gScene.materials, v, lod);
            if (opacity >= 0.5)
                return 0.0; // alpha test: hit opaque => occluded
#endif

        }
        else
            return 0.0; // hit opaque => occluded
    }

    if (rayQuery.CommittedStatus() != COMMITTED_NOTHING)  visibility = 0.0; // hit opaque => occluded

    // scale based on the cutoff
    visibility = saturate((visibility - visibilityCutoff) / (1.0 - visibilityCutoff));
    
    return visibility;
}

float main(float2 texC : TEXCOORD, float4 svPos : SV_POSITION) : SV_Target
{
    float4 posW = gPos[int2(svPos.xy)];
    if (posW.w == 0.0)
        return 1.0; // background
    
    if (gLightIndex >= gScene.getLightCount())
        return 1.0;

    float3 normalW = gNormal[int2(svPos.xy)].xyz;
    
    LightData light = gScene.getLight(gLightIndex);
    RayDesc ray;
    ray.Origin = posW.xyz;
    ray.TMin = gScene.camera.data.nearZ * 0.1;

    float3 V = gScene.camera.getPosition() - posW.xyz;
    float lenV = length(V);

    #if USE_RAYCONES
    RayCone rc = RayCone(0.0, RAY_CONE_SPREAD);
    rc = rc.propagateDistance(lenV); // travel t
    rc = rc.hit(V / lenV, normalW); // hit surface
    #endif
    
    float secondarySpreadAngle = RAY_CONE_SPREAD;
    if (light.type == uint(LightType::Point))
    {
        ray.Direction = normalize(light.posW - posW.xyz);
        ray.TMax = distance(light.posW, posW.xyz) - gPointLightClip;

        #if USE_RAYCONES
        rc = rc.reflect(ray.Direction, normalW);
        rc = rc.focus(distance(light.posW, posW.xyz));
        #endif
    }
    else if (light.type == uint(LightType::Directional))
    {
        ray.Direction = -light.dirW;
        ray.TMax = gScene.camera.data.farZ;
        secondarySpreadAngle = 0.0; // light rays are orthogonal

        #if USE_RAYCONES
        rc = rc.reflect(ray.Direction, normalW);
        rc = rc.orthogonalize();
        #endif
    }

    if (dot(ray.Direction, normalW) <= 0.0)
        return 0.0; // backfacing surface

    #if USE_RAYCONES
    float visibility = traceShadowRay(ray, rc);
    #else
    float visibility = traceShadowRay(ray);
    #endif
    
    return visibility;
}
