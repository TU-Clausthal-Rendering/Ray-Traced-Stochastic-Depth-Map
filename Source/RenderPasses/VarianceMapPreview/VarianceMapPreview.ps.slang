Texture2D gInput;
SamplerState S;

cbuffer CBuffer
{
    uint mode;
    uint level;
};

#define MODE_DEPTH 0
#define MODE_NORMAL 1

float4 main(float2 uv : TEXCOORD) : SV_TARGET
{
    float4 c = gInput.SampleLevel(S, uv, float(level));

    if(mode == MODE_DEPTH)
    {
        c.rgb = (c.y - c.x * c.x) / c.x;
        c.a = 1.0;
    }
    else if(mode == MODE_NORMAL)
    {
        float vx = c.y - c.x * c.x;
        float vy = c.w - c.z * c.z;
        c.rgb = vx + vy;
        c.a = 1.0;
    }

    return c;
}
