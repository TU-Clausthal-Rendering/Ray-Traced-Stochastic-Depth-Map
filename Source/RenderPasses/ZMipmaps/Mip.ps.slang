#include "Utils/Math/MathConstants.slangh"

Texture2D gSrc;
SamplerState S;

cbuffer CameraCB
{
    float farZ; // farplane
    float threshold; // threshold for depth test
}

float4 main(float2 uv : TEXCOORD) : SV_TARGET
{
    #if 0
    // default radius
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

    #else
    // big radius
    float2 dudv = 0.5 * (ddx(uv) + ddy(uv));

    float4x4 gathers;
    gathers[0] = gSrc.Sample(S, uv + float2(-dudv.x, -dudv.y));
    gathers[1] = gSrc.Sample(S, uv + float2(+dudv.x, -dudv.y));
    gathers[2] = gSrc.Sample(S, uv + float2(-dudv.x, +dudv.y));
    gathers[3] = gSrc.Sample(S, uv + float2(+dudv.x, +dudv.y));

    #define d(i) (gathers[i / 4u][i % 4u])
    
    float ref = 1e-10;
    [unroll] for (uint i = 0; i < 16; ++i)
        if (d(i) > ref && d(i) < farZ) // max, but not farZ
            ref = d(i);

    float4x4 wgather; // set weights
    #define w(i) (wgather[i / 4u][i % 4u])
    
    [unroll]
    for (uint i = 0; i < 16; ++i)
    {
        w(i) = 0.0;
        if (d(i) > threshold * ref && d(i) < farZ)
            w(i) = 1.0;
    }

    if (all(wgather == 0.0))
        return farZ; // all farplane

    // normalize weights
    float wsum = 0.0;
    [unroll] for (uint i = 0; i < 16; ++i)
        wsum += w(i);

    wgather = wgather / wsum;

    // weighted average
    float res = 0.0;
    [unroll] for (uint i = 0; i < 16; ++i)
        //res += w(i) * d(i);
        res += w(i) * rcp(max(d(i), 1e-7));

    return rcp(res); // perspective correct interpolation 1/z
    //return res;
    
    #endif
}
