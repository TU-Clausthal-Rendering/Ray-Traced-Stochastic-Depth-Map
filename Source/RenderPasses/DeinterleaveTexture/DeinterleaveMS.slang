#ifndef type
#define type float
#endif

Texture2DMS<type> src;

cbuffer PassData
{
    uint offset;
};

struct PsOut
{
    type target0 : SV_Target0;
    type target1 : SV_Target1;
    type target2 : SV_Target2;
    type target3 : SV_Target3;
    type target4 : SV_Target4;
    type target5 : SV_Target5;
    type target6 : SV_Target6;
    type target7 : SV_Target7;
};

uint2 calcPosFromOffset(uint off)
{
    return uint2(off % 4, off / 4);
}

PsOut main(float2 uv : TEXCOORD, float4 svPos : SV_POSITION, uint sampleIndex : SV_SampleIndex) : SV_TARGET
{
    PsOut o;
    uint2 xyQuarter = uint2(svPos.xy); // position in quarter resolution texture
    uint2 xy = xyQuarter * 4u; // position in full res texture

    o.target0 = src.Load(xy + calcPosFromOffset(offset + 0), sampleIndex);
    o.target1 = src.Load(xy + calcPosFromOffset(offset + 1), sampleIndex);
    o.target2 = src.Load(xy + calcPosFromOffset(offset + 2), sampleIndex);
    o.target3 = src.Load(xy + calcPosFromOffset(offset + 3), sampleIndex);
    o.target4 = src.Load(xy + calcPosFromOffset(offset + 4), sampleIndex);
    o.target5 = src.Load(xy + calcPosFromOffset(offset + 5), sampleIndex);
    o.target6 = src.Load(xy + calcPosFromOffset(offset + 6), sampleIndex);
    o.target7 = src.Load(xy + calcPosFromOffset(offset + 7), sampleIndex);

    return o;
}
