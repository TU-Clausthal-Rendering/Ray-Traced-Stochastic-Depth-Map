import Scene.Camera.Camera;

Texture2D<float> gLinearDepth;

cbuffer PerFrameCB
{
    Camera gCamera;
}

// uv: uv coordinates [0, 1]
// viewDepth: linear depth in view space (positive z)
// return: view space position (negative z)
float3 UVToViewSpace(float2 uv, float viewDepth)
{
    float2 ndc = float2(uv.x, 1.0 - uv.y) * 2.0 - 1.0; // normalized device coordinates [-1, 1]
    const float2 imageScale = 0.5 * float2(gCamera.data.frameWidth / gCamera.data.focalLength, gCamera.data.frameHeight / gCamera.data.focalLength);
    return float3(ndc * viewDepth * imageScale, -viewDepth);
}

// returns [0, 1] clamped uv coordinate matching the pixel
float2 PixelToUV(int2 pixel)
{
    float width, height;
    gLinearDepth.GetDimensions(width, height);
    return saturate((float2(pixel) + 0.5) / float2(width, height));
}

float RelativeDepth(float reference, float value)
{
    return abs(1.0 - value / reference);
}

float AbsoluteDepth(float reference, float value)
{
    return abs(reference - value);
}

uint main(float2 uv : TEXCOORD, float4 svPos : SV_POSITION) : SV_TARGET
{
    int2 px = int2(svPos.xy);
    float P = gLinearDepth[px];
    float P_right = gLinearDepth[px + int2(1, 0)];
    float P_left = gLinearDepth[px - int2(1, 0)];
    float P_down = gLinearDepth[px + int2(0, 1)];
    float P_up = gLinearDepth[px - int2(0, 1)];

    // min distance of neighbors
    /*float dx = min(RelativeDepth(P, P_right), RelativeDepth(P, P_left));
    float dy = min(RelativeDepth(P, P_down), RelativeDepth(P, P_up));

    float threshold = 0.001;
    //if (dx <= threshold && dy <= threshold)
    if(min(dx, dy) < threshold)
        return 1;
    return 0;*/
    float dx = min(AbsoluteDepth(P, P_right), AbsoluteDepth(P, P_left));
    float dy = min(AbsoluteDepth(P, P_down), AbsoluteDepth(P, P_up));
    float threshold = gCamera.data.nearZ * 0.4;
    if(dx <= threshold && dy <= threshold)
        return 1;
    return 0;
    
    /*float3 P = UVToViewSpace(uv, gLinearDepth[px]);
    // reconstruct nearby view depths
    float3 P_right = UVToViewSpace(PixelToUV(px + int2(1, 0)), gLinearDepth[px + int2(1, 0)]);
    float3 P_left = UVToViewSpace(PixelToUV(px + int2(-1, 0)), gLinearDepth[px + int2(-1, 0)]);
    float3 P_top = UVToViewSpace(PixelToUV(px + int2(0, 1)), gLinearDepth[px + int2(0, 1)]);
    float3 P_bot = UVToViewSpace(PixelToUV(px + int2(0, -1)), gLinearDepth[px + int2(0, -1)]);

    // min distance of neighbors
    float dx = min(distance(P, P_right), distance(P, P_left));
    float dy = min(distance(P, P_top), distance(P, P_bot));

    float threshold = distance(P, UVToViewSpace(PixelToUV(px + int2(1, 0)), gLinearDepth[px])) * 4.0; // as reference, take difference to neighbor pixel with same depth

    if (dx <= threshold && dy <= threshold)
        return 1u; // continous => no flicker
    return 0u; // potential flicker*/

}
