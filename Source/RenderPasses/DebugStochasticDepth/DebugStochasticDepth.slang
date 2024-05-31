import Scene.Camera.Camera;

Texture2DArray<float4> gsDepthTex;

cbuffer PerFrameCB
{
    Camera gCamera;
    float4x4 prevViewToCurView;
    float4x4 newProjection;
    uint2 resolution; // screen resolution
    uint numSamples;
    uint sdGuard; // guard band size
    uint sdJitter; // 1 if jitter enabled
}

struct VSOut
{
    bool valid : VALID;
    float depth : DEPTH;
    float4 pos : SV_Position;
};

static const float2 jitterPos[16] = { float2(0.125, 0.125), float2(0.375, 0.875), float2(0.625, 0.625), float2(0.875, 0.375), float2(0.875, 0.375), float2(0.625, 0.625), float2(0.375, 0.875), float2(0.125, 0.125), float2(0.625, 0.625), float2(0.875, 0.375), float2(0.125, 0.125), float2(0.375, 0.875), float2(0.375, 0.875), float2(0.125, 0.125), float2(0.875, 0.375), float2(0.625, 0.625) };

float2 randomJitter(uint2 pixel)
{
    if (sdJitter)
    {
        uint2 block = pixel / 4u;
        pixel = pixel % 4u;
        uint index = pixel.y * 4 + pixel.x;
        
        return jitterPos[index];
    }
    return float2(0.5);
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

VSOut vsMain(uint id : SV_VertexID)
{
    uint k = id / (resolution.x * resolution.y); // stocastic depth k
    uint2 pixel = uint2(id % resolution.x, (id / resolution.x) % resolution.y);

    float2 sdSampleUV = (pixel - sdGuard + randomJitter(pixel)) / float2(resolution - 2u * sdGuard);

    float4 sddepths[2];
    sddepths[0] = gsDepthTex[uint3(pixel, 0)];
    if(numSamples > 4) sddepths[1] = gsDepthTex[uint3(pixel, 1)];
    
    const float depthRange = gCamera.data.farZ - gCamera.data.nearZ;
    const float depthOffset = gCamera.data.nearZ;
    float rawDepth = sddepths[k / 4][k % 4];
    float linearSampleDepth = rawDepth * depthRange + depthOffset;

    // view pos with sd map camera
    float3 samplePosV = UVToViewSpace(sdSampleUV, linearSampleDepth);
    float3 samplePosW = mul(prevViewToCurView, float4(samplePosV, 1.0)).xyz;
    
    VSOut o;
    o.pos = mul(newProjection, float4(samplePosW, 1.0));
    //o.pos = float4(sdSampleUV * 2.0 - 1.0, 0.0, 1.0);
    //o.pos = float4(0.0, 0.0, 0.0, 1.0);
    o.valid = rawDepth < 1.0 && rawDepth > 0.0;
    o.depth = linearSampleDepth;
    return o;
}

float4 psMain(bool valid : VALID, float depth : DEPTH) : SV_TARGET
{
    if(!valid)
        discard;
    
    return float4(depth, depth / 10.0, depth / 100.0, 1.0);
}
