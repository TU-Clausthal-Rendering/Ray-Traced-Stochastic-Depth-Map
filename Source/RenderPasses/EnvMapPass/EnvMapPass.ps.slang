import Scene.Raster;

float3 main(float2 uv : TEXCOORD, float4 svPos : SV_POSITION) : SV_TARGET
{
    uint2 pixel = int2(svPos.xy);
    // get camera ray from pixel
    let data = gScene.camera.data;
    float2 ndc = float2(2, -2) * uv + float2(-1, 1);
    float3 dir = ndc.x * data.cameraU + ndc.y * data.cameraV + data.cameraW;
    dir = normalize(dir);

    // lookup dir in envmap
    float3 color = gScene.envMap.eval(dir);
    return color;
}
