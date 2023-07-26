import Scene.Camera.Camera;

#ifndef MAX_ITERATION
#define MAX_ITERATION 5
#endif

Texture2D<float2> gMotionVec;
Texture2D<float> gDepth;
Texture2D<float> gPrevDepth;
Texture2D<float> gPrevDepth2;

SamplerState gDepthSampler;

cbuffer PerFrameCB
{
    Camera gCamera;
    float4x4 prevViewToCurView; // viewMat * Inverse(prevViewMat)
    float4x4 curViewToPrevView; // prevViewMat * Inverse(viewMat)
}

bool isInValidArea(float2 uv)
{
    return uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0;
}

int2 UVToPixel(float2 uv)
{
    float width, height;
    gDepth.GetDimensions(width, height);
    return int2(floor(uv * float2(width, height)));
}

#define INSIDE 0
#define LEFT 1
#define RIGHT 2
#define BOTTOM 4
#define TOP 8

uint ComputeRegionCode(float u, float v)
{
    uint code = INSIDE;

    if (u < 0.0f)
        code |= LEFT;
    else if (u > 1.0f)
        code |= RIGHT;

    if (v < 0.0f)
        code |= BOTTOM;
    else if (v > 1.0f)
        code |= TOP;

    return code;
}

bool CohenSutherlandClipping(inout float2 p0, inout float2 p1)
{
    uint code0 = ComputeRegionCode(p0.x, p0.y);
    uint code1 = ComputeRegionCode(p1.x, p1.y);

    while (true) // max number of iterations = 4
    {
        if ((code0 | code1) == 0)
        {
            // Both points are inside the clipping window, the line is fully visible
            return true;
        }
        else if ((code0 & code1) != 0)
        {
            // Both points are outside the same region, and the line is fully outside the window
            // In this case, we discard the line
            return false;
        }
        else
        {
            // The line is partially visible, we need to clip it

            // Select one of the points outside the window
            uint codeOutside = (code0 != 0) ? code0 : code1;

            float u, v; // Intersection point with the clipping window
            float2 p;

            // Find the intersection point with the corresponding edge of the window
            if ((codeOutside & TOP) != 0)
            {
                u = p0.x + (p1.x - p0.x) * (1.0f - p0.y) / (p1.y - p0.y);
                v = 1.0f;
            }
            else if ((codeOutside & BOTTOM) != 0)
            {
                u = p0.x + (p1.x - p0.x) * (0.0f - p0.y) / (p1.y - p0.y);
                v = 0.0f;
            }
            else if ((codeOutside & RIGHT) != 0)
            {
                v = p0.y + (p1.y - p0.y) * (1.0f - p0.x) / (p1.x - p0.x);
                u = 1.0f;
            }
            else if ((codeOutside & LEFT) != 0)
            {
                v = p0.y + (p1.y - p0.y) * (0.0f - p0.x) / (p1.x - p0.x);
                u = 0.0f;
            }

            // Update the point outside the window with the intersection point
            if (codeOutside == code0)
            {
                p0.x = u;
                p0.y = v;
                code0 = ComputeRegionCode(p0.x, p0.y);
            }
            else
            {
                p1.x = u;
                p1.y = v;
                code1 = ComputeRegionCode(p1.x, p1.y);
            }
        }
    }
}



// uv: uv coordinates [0, 1]
// viewDepth: linear depth in view space (positive z)
// return: view space position (negative z)
float3 UVToViewSpace(float2 uv, float viewDepth)
{
    float2 ndc = float2(uv.x, 1.0 - uv.y) * 2.0 - 1.0; // normalized device coordinates [-1, 1]
    const float2 imageScale = 0.5 * float2(gCamera.data.frameWidth / gCamera.data.focalLength, gCamera.data.frameHeight / gCamera.data.focalLength);
    return float3(ndc * viewDepth * imageScale, -viewDepth);
}

// posV: view space position (negative z)
// return: texture uv [0, 1]
float2 ViewSpaceToUV(float3 posV)
{
    const float2 imageScale = 0.5 * float2(gCamera.data.frameWidth / gCamera.data.focalLength, gCamera.data.frameHeight / gCamera.data.focalLength);
    float2 ndc = posV.xy / (imageScale * posV.z);
    return ndc * float2(-0.5, 0.5) + 0.5; // since posV.z is negative, the sign order is inversed
}

float RelativeDepth(float reference, float value)
{
    return abs(1.0 - value / reference);
}

float CalcViewDepthFromPrev(float2 prevUV, Texture2D<float> tex)
{
    float3 prevPosV = UVToViewSpace(prevUV, tex.SampleLevel(gDepthSampler, prevUV, 0.0));
    float3 curPosV = mul(prevViewToCurView, float4(prevPosV, 1.0)).xyz;
    return -curPosV.z;
}

struct PsOut
{
    float depth2 : SV_Target0;
};

float maxComponent(float4 v)
{
    //return max(max(v.x, v.y), max(v.z, v.w));
    return (v.x + v.y + v.z + v.w) * 0.25;
}

PsOut main(float2 texC : TEXCOORD, float4 svPos : SV_POSITION)
{
    float depth = gDepth.SampleLevel(gDepthSampler, texC, 0.0);
    PsOut o;
    o.depth2 = depth;
    //o.depth2 = gCamera.data.farZ; // assume far plane
    
    float2 mvec = gMotionVec.SampleLevel(gDepthSampler, texC, 0.0);
    
    if (!isInValidArea(texC + mvec)) // test if motion vector is outside of the screen
    {
        return o; // no information
    }

    float3 curPosV = UVToViewSpace(texC, depth);

    float2 uvmin = texC + mvec;
    //int2 prevPixel = UVToPixel(prevUV);

    float3 curPosVMax = UVToViewSpace(texC, gCamera.data.farZ);
    float2 uvmax = ViewSpaceToUV(mul(curViewToPrevView, float4(curPosVMax, 1.0)).xyz);

    float2 minUVvec = ViewSpaceToUV(mul(prevViewToCurView, float4(UVToViewSpace(uvmin, gCamera.data.farZ), 1.0)).xyz) - texC;

    CohenSutherlandClipping(uvmin, uvmax);
    
    // init best result so far
    float bestUVDist = 10000.0;
    o.depth2 = depth;
    
    // do binary search for the correct depth value
    for (int i = 0; i < MAX_ITERATION; ++i)
    {
        float2 uvmid = (uvmin + uvmax) * 0.5;
        
        // obtain previous depth values at uvmid
        float d[2];
        d[0] = gPrevDepth.SampleLevel(gDepthSampler, uvmid, 0.0);
        d[1] = gPrevDepth2.SampleLevel(gDepthSampler, uvmid, 0.0);
        //d[0] = maxComponent(gPrevDepth.Gather(gDepthSampler, uvmid));
        //d[1] = maxComponent(gPrevDepth2.Gather(gDepthSampler, uvmid));
        d[1] = max(d[0], d[1]); // d[1] can be zero if it was invalid

        float2 curUV;
        // check both depth values
        for (int di = 0; di < 2; ++di)
        {
            // calc uv in current view space
            float3 prevPosV = UVToViewSpace(uvmid, d[di]);
            float3 curPosV = mul(prevViewToCurView, float4(prevPosV, 1.0)).xyz;
            curUV = ViewSpaceToUV(curPosV);

            float uvDist = dot(curUV, texC);
            if (uvDist < bestUVDist && -curPosV.z > depth + 0.2)
            //if (uvDist < bestUVDist && -curPosV.z > depth * 1.1)
            {
                o.depth2 = -curPosV.z; // ~ d[di]
                bestUVDist = uvDist;
            }
        }

        // update uvmid
        if (dot(minUVvec, curUV) > 0.0) uvmax = uvmid;
        else uvmin = uvmid;
    }
    
    return o;
}
