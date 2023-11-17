import Scene.Raster;
import Utils.Sampling.TinyUniformSampleGenerator;
import Rendering.Lights.LightHelpers;
import Rendering.Materials.Fresnel;

Texture2DArray<float> visibilityBuffer;

cbuffer ConstantCB
{
    float gAmbientIntensity;
    float gEnvMapIntensity;
    float gLightIntensity;
    bool gEnvMapMirror;
};

VSOut vsMain(VSIn vIn)
{
    return defaultVS(vIn);
}

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

float4 psMain(VSOut vsOut, uint triangleIndex : SV_PrimitiveID) : SV_TARGET
{
    let lod = ImplicitLodTextureSampler();
    if (alphaTest(vsOut, triangleIndex, lod))
        discard;

    uint2 pixel = uint2(vsOut.posH.xy);

    float3 viewDir = normalize(gScene.camera.getPosition() - vsOut.posW);
    ShadingData sd = prepareShadingData(vsOut, triangleIndex, viewDir, lod);
    
    // Create material instance.
    let mi = gScene.materials.getMaterialInstance(sd, lod, (uint)MaterialInstanceHints::AdjustShadingNormal);
    let bsdf = mi.getProperties(sd);
    
    // emission
    float3 color = bsdf.emission;
    // constant ambient term
    color += gAmbientIntensity * bsdf.diffuseReflectionAlbedo;
    // envmap ambient term
    float3 R = reflect(-viewDir, bsdf.guideNormal);
    float diffLod = roughnessToLod(bsdf.roughness * 2.0);
    float specLod = roughnessToLod(bsdf.roughness);
    float envMaxIntensity = roughnessToIntensity(bsdf.roughness);
    float F = evalFresnelSchlick(0.3, 1.0, max(0.0, dot(viewDir, bsdf.guideNormal)));
    color += gEnvMapIntensity * bsdf.diffuseReflectionAlbedo * min(gScene.envMap.eval(lerp(sd.frame.N, sd.faceN, 0.8), diffLod), 1.0);
    color +=  gEnvMapIntensity * bsdf.specularReflectionAlbedo * min(gScene.envMap.eval(R, specLod), envMaxIntensity);

    // perfect mirror
    if (gEnvMapMirror && bsdf.roughness < 0.08)
        color += F * gScene.envMap.eval(R, specLod);
    
    TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(pixel, 0);

    uint visWidth, visHeight, visSlices;
    visibilityBuffer.GetDimensions(visWidth, visHeight, visSlices);
    
    // Direct lighting from analytic light sources
    for (int i = 0; i < gScene.getLightCount(); i++)
    {
        AnalyticLightSample ls;
        evalLightApproximate(sd.posW, gScene.getLight(i), ls);
        float shadow = 1.0;
        if(i < visSlices)
            shadow = visibilityBuffer[uint3(pixel, i)];

        
        color += shadow * mi.eval(sd, ls.dir, sg) * ls.Li * gLightIntensity;
        
    }

    return float4(color, 1.f);
}
