Texture2D gSrc;
SamplerState S;

cbuffer CameraCB
{
    float farZ; // farplane
    float threshold; // threshold for depth test
}

float4 main(float2 uv : TEXCOORD) : SV_TARGET
{
    float4 d = gSrc.Gather(S, uv);
    float ref = 1e-10;
    //ref = max(max(d[0], d[1]), max(d[2], d[3])); // max
    [unroll] for (uint i = 0; i < 4; ++i)
        if(d[i] > ref && d[i] < farZ) // max, but not farZ
            ref = d[i];

    float4 w = 0.0; // set weights
    [unroll] for (uint i = 0; i < 4; ++i)
    {
        w[i] = 0.0;
        if (d[i] > threshold * ref && d[i] < farZ)
            w[i] = 1.0;
    }

    if (all(w == 0.0))
        return farZ; // all farplane

    // weighted average
    //return dot(w, d) / (w[0] + w[1] + w[2] + w[3]);
    w = w / (w[0] + w[1] + w[2] + w[3]);
    return rcp(dot(w, rcp(d))); // perspective correct interpolation 1/z
}
