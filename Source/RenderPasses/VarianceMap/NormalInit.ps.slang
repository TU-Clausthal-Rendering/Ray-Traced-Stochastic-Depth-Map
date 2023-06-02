Texture2D<float4> normalTex;

cbuffer PerFrameCB
{
    float4x4 viewMat;
}

SamplerState S;

float4 main(float2 uv : TEXCOORD) : SV_TARGET
{
    float4 normalW = normalTex.Sample(S, uv);
    normalW.w = 0.0; // make sure this is zero
    float3 normalV = mul(viewMat, normalW).xyz;
    return float4(normalV.x, normalV.x * normalV.x, normalV.y, normalV.y * normalV.y);
}
