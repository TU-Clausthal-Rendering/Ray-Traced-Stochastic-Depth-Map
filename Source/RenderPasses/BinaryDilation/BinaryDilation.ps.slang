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
    float2 duv = 0.5f / dim; // half pixel diff
    //uv = floor(svPos.xy) * invDim;

    uint4 res;

    // small radius
    //res.x = Reduce(gInput.Gather(S, uv + float2(-duv.x, -duv.y)));
    //res.y = Reduce(gInput.Gather(S, uv + float2(duv.x, -duv.y)));
    //res.z = Reduce(gInput.Gather(S, uv + float2(-duv.x, duv.y)));
    //res.w = Reduce(gInput.Gather(S, uv + float2(duv.x, duv.y)));

    // bigger radius
    res.x = Reduce(gInput.Gather(S, uv + float2(duv.x, 3.0 * duv.y)));
    res.y = Reduce(gInput.Gather(S, uv + float2(3.0 * duv.x, -duv.y)));
    res.z = Reduce(gInput.Gather(S, uv + float2(-duv.x, -3.0 * duv.y)));
    res.w = Reduce(gInput.Gather(S, uv + float2(-3.0 * duv.x, duv.y)));
    // add center
    res.x = OP(res.x, Reduce(gInput.Gather(S, uv)));
    
    uint o = Reduce(res);
    //o = WaveActiveBitAnd(o);

    return o;
}
