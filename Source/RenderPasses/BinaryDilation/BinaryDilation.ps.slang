Texture2D<uint> gInput;
SamplerState S;

#ifndef OP
#define OP min
#endif

uint Reduce(uint4 value)
{
    return OP(OP(value.x, value.y), OP(value.z, value.w));
}

uint main(float2 uv : TEXCOORD, float4 svPos : SV_POSITION) : SV_TARGET
{
    // obtain screen size
    float2 dim;
    gInput.GetDimensions(dim.x, dim.y);
    float2 invDim = 1.0f / dim;
    uv = floor(svPos.xy) * invDim;

    uint4 res;
    res.x = Reduce(gInput.Gather(S, uv));
    res.y = Reduce(gInput.Gather(S, uv + float2(invDim.x, 0.0)));
    res.z = Reduce(gInput.Gather(S, uv + float2(0.0, invDim.y)));
    res.w = Reduce(gInput.Gather(S, uv + invDim));

    return Reduce(res);
}
