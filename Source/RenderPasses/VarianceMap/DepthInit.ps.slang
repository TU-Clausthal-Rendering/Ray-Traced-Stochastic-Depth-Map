Texture2D<float> depthTex;

SamplerState S;

float2 main(float2 uv : TEXCOORD) : SV_TARGET
{
    float d = depthTex.Sample(S, uv);
    return float2(d, d * d);
}
