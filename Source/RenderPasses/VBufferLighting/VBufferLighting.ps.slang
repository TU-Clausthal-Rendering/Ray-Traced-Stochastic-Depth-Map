import Scene.Raster;
import Utils.Sampling.TinyUniformSampleGenerator;
import Rendering.Lights.LightHelpers;
import Scene.Intersection;
import Utils.Math.MathHelpers;
import Scene.Intersection;
import Rendering.Materials.Fresnel;

Texture2D<float> visibilityBuffer;
Texture2D<PackedHitInfo> vbuffer;

cbuffer ConstantCB
{
    float gAmbientIntensity;
    float gEnvMapIntensity;
};

float roughnessToLod(float roughness)
{
    float nEnvLods = log2(float(gScene.envMap.getDimensions().x));
    float maxEnvLod = max(nEnvLods - 4.0, 0.0);
    return saturate(roughness) * maxEnvLod;
}

// limit maximal intensity of rough materials (avoids bright sparkles)
float roughnessToIntensity(float roughness)
{
    return saturate(1.0 / (roughness + 0.1));
}

    /** Ray differentials for primary hit. Code from RayTracingGems, Chapter 20.
    */
void computeRayDifferentials(const TriangleHit hit, float3 rayDir, float hitT, const Camera camera, float2 frameDim, out float2 ddx, out float2 ddy)
{
    // Ray differentials
    float3 P[3];
    gScene.getVertexPositionsW(hit.instanceID, hit.primitiveIndex, P);
    float3 e1 = P[1] - P[0];
    float3 e2 = P[2] - P[0];
    float3 d = rayDir;
    float k = dot(cross(e1, e2), d);
    k = abs(k) > 1e-20f ? rcp(k) : 0.0f;
    float3 cu = cross(e2, d);
    float3 cv = cross(d, e1);
        // Assumes a normalized ray direction
    float3 dx = camera.data.cameraU * 2.f / (camera.data.focalDistance * frameDim.x); // dDdx in ray gen
    float3 dy = camera.data.cameraV * 2.f / (camera.data.focalDistance * frameDim.y); // dDdy in ray gen
    float3 q = dx * hitT; // Transfer to primary hit
    float3 r = dy * hitT;
    float dudx = k * dot(cu, q);
    float dudy = k * dot(cu, r);
    float dvdx = k * dot(cv, q);
    float dvdy = k * dot(cv, r);
    float2 T[3];
    gScene.getVertexTexCoords(hit.instanceID, hit.primitiveIndex, T);
    float2 g1 = T[1] - T[0];
    float2 g2 = T[2] - T[0];
    float dsdx = (dudx * g1.x + dvdx * g2.x);
    float dsdy = (dudy * g1.x + dvdy * g2.x);
    float dtdx = (dudx * g1.y + dvdx * g2.y);
    float dtdy = (dudy * g1.y + dvdy * g2.y);
    ddx = float2(dsdx, dtdx);
    ddy = float2(dsdy, dtdy);
}

float4 main(float2 texC : TEXCOORD, float4 svPos : SV_POSITION) : SV_TARGET
{
    // pixel coordinates
    uint2 pixel = uint2(svPos.xy);
    uint2 frameDim;
    vbuffer.GetDimensions(frameDim.x, frameDim.y);
    
    const HitInfo hit = HitInfo(vbuffer[pixel]);

    float3 color = 0.0;
    if (hit.isValid()) // valid hit
    {
        // Load shading data.
        float3 viewDir = -gScene.camera.computeRayPinhole(pixel, frameDim).dir;
        VertexData v = { };
        uint materialID = { };
        if (hit.getType() != HitType::Triangle)
            return 0.0;
        
        const TriangleHit triangleHit = hit.getTriangleHit();
        v = gScene.getVertexData(triangleHit);
        materialID = gScene.getMaterialID(triangleHit.instanceID);

        // ray differentials: (Implicit does not work with vbuffer, because of different triangles per pixels)
        float2 ddx, ddy;
        float hitT = distance(v.posW, gScene.camera.data.posW);
        computeRayDifferentials(triangleHit, -viewDir, hitT, gScene.camera, float2(frameDim), ddx, ddy);
        let lod = ExplicitGradientTextureSampler(ddx, ddy);
        //let lod = ImplicitLodTextureSampler();
        //let lod = ExplicitLodTextureSampler(1000.0);
        
        
        ShadingData sd = gScene.materials.prepareShadingData(v, materialID, viewDir, lod);

        // Create material instance at shading point.
        let mi = gScene.materials.getMaterialInstance(sd, lod, (uint) MaterialInstanceHints::AdjustShadingNormal);
    
        // Create material instance.
        let bsdf = mi.getProperties(sd);
    
        // emission
        color = bsdf.emission;
        // constant ambient term
        color += gAmbientIntensity * bsdf.diffuseReflectionAlbedo;
        // envmap ambient term
        float3 R = reflect(-viewDir, bsdf.guideNormal);
        float diffLod = roughnessToLod(bsdf.roughness * 2.0);
        float specLod = roughnessToLod(bsdf.roughness);
        float envMaxIntensity = roughnessToIntensity(bsdf.roughness);
        float F = evalFresnelSchlick(0.3, 1.0, max(0.0, dot(viewDir, bsdf.guideNormal)));
        color += gEnvMapIntensity * bsdf.diffuseReflectionAlbedo * min(gScene.envMap.eval(lerp(sd.frame.N, sd.faceN, 0.8), diffLod), 1.0);
        color += F * gEnvMapIntensity * bsdf.specularReflectionAlbedo * min(gScene.envMap.eval(R, specLod), envMaxIntensity);

        // perfect mirror
        if(bsdf.roughness < 0.08)
            color += F * gScene.envMap.eval(R, specLod);
        
        TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(pixel, 0);

        // Direct lighting from analytic light sources
        for (int i = 0; i < gScene.getLightCount(); i++)
        {
            AnalyticLightSample ls;
            evalLightApproximate(sd.posW, gScene.getLight(i), ls);
            float shadow = 1.0;
            if (i == 0)
                shadow = visibilityBuffer[pixel];

        
            color += shadow * mi.eval(sd, ls.dir, sg) * ls.Li;
        
        }

        return float4(color, 1.f);
    }
    else // invalid hit => envmap
    {
        // get camera ray from pixel
        let data = gScene.camera.data;
        float2 ndc = float2(2, -2) * texC + float2(-1, 1);
        float3 dir = ndc.x * data.cameraU + ndc.y * data.cameraV + data.cameraW;
        dir = normalize(dir);

        // lookup dir in envmap
        color = gScene.envMap.eval(dir) * gEnvMapIntensity;
    }
    return float4(color, 1.f);
}
