Texture2D<float> gDepth;
Texture2D<float> gDepth2;

//SamplerState gLinearSampler;
//SamplerState gPointSampler;

float RelativeDepth(float reference, float value)
{
    return abs(1.0 - value / reference);
}

struct PsOut
{
    float depth2 : SV_Target0;
};

//[earlydepthstencil] // not required to be forced
PsOut main(float2 texC : TEXCOORD, float4 svPos : SV_POSITION)
{
    int2 offsets[] = { int2(1, 0), int2(-1, 0), int2(0, 1), int2(0, -1) };
    
    int2 pixel = int2(svPos.xy);
    // if the stencil test passes, this depth will be a value from the current frame => try to interplate by taking values from the previous frame 
    float depth = gDepth2[pixel];
    float neighbors[4];
    [unroll] for (int i = 0; i < 4; ++i)
    {
        neighbors[i] = gDepth2[pixel + offsets[i]];
    }

    float newDepth = 0.0;
    
    float xAvg = (neighbors[0] + neighbors[1]) * 0.5;
    float yAvg = (neighbors[2] + neighbors[3]) * 0.5;
    bool useX = RelativeDepth(neighbors[0], neighbors[1]) < 0.1;
    bool useY = RelativeDepth(neighbors[2], neighbors[3]) < 0.1;
    if(useX && useY)
        newDepth = max(xAvg, yAvg);
    else if(useX)
        newDepth = xAvg;
    else if(useY)
        newDepth = yAvg;
   

    // overwrite with interpolated depth
    if (newDepth <= depth + 0.4)
        discard;

    PsOut o;
    o.depth2 = newDepth;
    return o;
}
