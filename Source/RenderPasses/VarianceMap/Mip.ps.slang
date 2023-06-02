Texture2D tex;
SamplerState S;

float4 main(float2 uv : TEXCOORD) : SV_TARGET
{
    return tex.Sample(S, uv);
}
