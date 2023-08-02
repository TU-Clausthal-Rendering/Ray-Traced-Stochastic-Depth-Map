import Scene.Camera.Camera;

Texture2D<float2> gMotionVec;
Texture2D<float> gDepth;
Texture2D<float> gPrevDepth;
Texture2D<float> gPrevDepth2;

SamplerState gLinearSampler;
SamplerState gPointSampler;

cbuffer PerFrameCB
{
    Camera gCamera;
    float4x4 prevViewToCurView; // viewMat * Inverse(prevViewMat)
    float4x4 curViewToPrevView; // prevViewMat * Inverse(viewMat)
    uint2 resolution;
    float minSeparationDist;
    int maxIterations;
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
    float3 prevPosV = UVToViewSpace(prevUV, tex.SampleLevel(gLinearSampler, prevUV, 0.0));
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

// perspective correct interpolation
float zlerp(float z0, float z1, float t)
{
    //return rcp(lerp(rcp(z0), rcp(z1), t));
    return 1.0 / lerp(1.0 / z0, 1.0 / z1, t);
}

float GetRectifiedDepth(float2 uv, Texture2D<float> depthTex)
{
    // TODO optimize this by only relying on n and doing the interpolation manually (currently 25% performance drop)
    float4 n = depthTex.Gather(gLinearSampler, uv, 0.0);
    float depth = depthTex.SampleLevel(gLinearSampler, uv, 0.0);
    float depthPoint = depthTex.SampleLevel(gPointSampler, uv, 0.0);
    //float minDist = min(min(abs(n[0] - depth), abs(n[1] - depth)), min(abs(n[2] - depth), abs(n[3] - depth)));
    //float minN = min(min(n[0], n[1]), min(n[2], n[3]));
    //float maxN = max(max(n[0], n[1]), max(n[2], n[3]));
    if(any(n > 0.99 * gCamera.data.farZ))
        return depthPoint;
    //if (maxN - minN > minSeparationDist && minDist > minSeparationDist * 0.01)
    //    return depthPoint;
    return depth;
}

PsOut main(float2 texC : TEXCOORD, float4 svPos : SV_POSITION)
{
    float depth = gDepth.SampleLevel(gLinearSampler, texC, 0.0);
    PsOut o;
    o.depth2 = depth;
    //o.depth2 = gCamera.data.farZ; // assume far plane
    
    /*float2 mvec = gMotionVec.SampleLevel(gLinearSampler, texC, 0.0);
    
    if (!isInValidArea(texC + mvec)) // test if motion vector is outside of the screen
    {
        return o; // no information
    }*/

    // calc min an max posV in current view space
    float3 minPosV = UVToViewSpace(texC, depth + minSeparationDist);
    float3 maxPosV = UVToViewSpace(texC, gCamera.data.farZ);
    // transform to previous view space
    minPosV = mul(curViewToPrevView, float4(minPosV, 1.0)).xyz;
    maxPosV = mul(curViewToPrevView, float4(maxPosV, 1.0)).xyz;
    // transform to uv coordinates in previous view space
    float2 minUV = ViewSpaceToUV(minPosV);
    float2 maxUV = ViewSpaceToUV(maxPosV);
    float minZ = -minPosV.z;
    float maxZ = -maxPosV.z;
    
    // clip to screen (uvmax may be outside)
    CohenSutherlandClipping(minUV, maxUV);

    float2 bestUV = 0.0;
    float bestZ = 0.0;
    
    float t = 0.0; // interpolation between min and max
    float tmin = 0.0;
    float tmax = 1.0;
    for (int i = 0; i < maxIterations; ++i)
    {
        t = (tmin + tmax) * 0.5;
        float2 uv = lerp(minUV, maxUV, t);
        float zRef = zlerp(minZ, maxZ, t); // reference z value of the current pixels ray
        // obtain rectified depth values from the previous frame
        float2 d;
        d[0] = GetRectifiedDepth(uv, gPrevDepth);
        d[1] = GetRectifiedDepth(uv, gPrevDepth2);
        d[1] = max(d[0], d[1]); // make sure d1 is not smaller than d0
        if (d[1] > 0.9 * gCamera.data.farZ)
            d[1] = d[0]; // dont use far plane

        // remember best (here last) values so far
        bestZ = d[1];
        bestUV = uv;
        
        if(abs(zRef - d[1]) < minSeparationDist * 0.001) break;
        else if (zRef < d[1]) // increase t
        {
            tmin = t;
        }
        else
        {
            tmax = t;
        }
    }
    
    /*for (int i = 0; i < maxIterations; ++i)
    {
        float2 uv = lerp(minUV, maxUV, t);
        float zRef = zlerp(minZ, maxZ, t); // reference z value of the current pixels ray
        // obtain rectified depth values from the previous frame
        float2 d;
        d[0] = GetRectifiedDepth(uv, gPrevDepth);
        d[1] = GetRectifiedDepth(uv, gPrevDepth2);
        d[1] = max(d[0], d[1]); // make sure d1 is not smaller than d0
        if (d[1] > 0.99 * gCamera.data.farZ)
            d[1] = d[0]; // dont use far plane

        if (zRef * 1.001 < d[1]) // increase t
        {
            t += 1.0f / maxIterations;
            // TODO do something if in d[0]
        }
        else
        {
            // take this point
            bestZ = zRef;
            bestUV = uv;
            break;
        }
    }*/

    if(bestZ == 0.0)
        return o; // no information

    // calc view position in previouse view space
    float3 bestPosV = UVToViewSpace(bestUV, bestZ);
    // cacl view position in current view space
    bestPosV = mul(prevViewToCurView, float4(bestPosV, 1.0)).xyz;
    // use z component as depth
    if(-bestPosV.z > depth + minSeparationDist)
        o.depth2 = -bestPosV.z;
    
    return o;
}
