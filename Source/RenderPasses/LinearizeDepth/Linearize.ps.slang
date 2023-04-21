Texture2D<float> depths;

SamplerState s;

cbuffer StaticCB
{
    float zNear;
    float zFar;
};

float4 main(float2 uv : TEXCOORD, float4 svPos : SV_POSITION) : SV_TARGET
{
    float nonLinear = depths.Sample(s, uv);
    float linearDepth = zNear * zFar / (zFar + nonLinear * (zNear - zFar));
    return linearDepth;
}
