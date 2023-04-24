import Scene.Raster;
import Utils.Sampling.TinyUniformSampleGenerator;
import Rendering.Lights.LightHelpers;

Texture2D<float> visibilityBuffer;

cbuffer ConstantCB
{
    float gAmbientIntensity;
    float gEnvMapIntensity;
};

VSOut vsMain(VSIn vIn)
{
    return defaultVS(vIn);
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
    let mi = gScene.materials.getMaterialInstance(sd, lod);

    // emission
    float3 color = mi.getProperties(sd).emission;
    // constant ambient term
    color += gAmbientIntensity *  mi.getProperties(sd).diffuseReflectionAlbedo;
    // envmap ambient term
    
    TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(pixel, 0);

    // Direct lighting from analytic light sources
    for (int i = 0; i < gScene.getLightCount(); i++)
    {
        AnalyticLightSample ls;
        evalLightApproximate(sd.posW, gScene.getLight(i), ls);
        float shadow = 1.0;
        if(i == 0)
            shadow = visibilityBuffer[pixel];

        
        color += shadow * mi.eval(sd, ls.dir, sg) * ls.Li;
        
    }

    return float4(color, 1.f);
}
