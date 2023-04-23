import Scene.Raster;
import Utils.Sampling.TinyUniformSampleGenerator;
import Rendering.Lights.LightHelpers;
import Rendering.Materials.BSDFs.DiffuseSpecularBRDF;

Texture2D<float4> gPos;
Texture2D<float4> gNorm;
Texture2D<float4> gDiff;
Texture2D<float4> gSpec;
Texture2D<float4> gEmissive;

float4 main(float2 uv : TEXCOORD, float4 svPos : SV_POSITION) : SV_TARGET
{
    float3 posW = gPos.Load(int3(svPos.xy, 0)).xyz;
    float3 normW = gNorm.Load(int3(svPos.xy, 0)).xyz;
    float4 diffOpacity = gDiff.Load(int3(svPos.xy, 0));
    float4 specRough = gSpec.Load(int3(svPos.xy, 0));
    float4 emissive = gEmissive.Load(int3(svPos.xy, 0));

    DiffuseSpecularBRDF brdf;
    brdf.diffuse = diffOpacity.rgb;
    brdf.specular = specRough.rgb;
    brdf.roughness = specRough.a;
    
    float4 color = float4(emissive.rgb, diffOpacity.a);

    // ambient light
    //color.rgb += 0.1 * diffOpacity.rgb;
    
    // iterate over all lights
    float3 viewDir = normalize(gScene.camera.getPosition() - posW);

    for (int i = 0; i < gScene.getLightCount(); i++)
    {
        AnalyticLightSample ls;
        evalLightApproximate(posW, gScene.getLight(i), ls);
        // TODO add shadow
        color.rgb += diffOpacity.rgb * ls.Li;
        //color.rgb += specRough.rgb * ls.Li;
        // TODO add diffuse specular brdf
        // add specular to color
        //color.rgb += brdf.evalSimple(ls.dir, viewDir) * ls.Li;
    }
    
    return color;
}
