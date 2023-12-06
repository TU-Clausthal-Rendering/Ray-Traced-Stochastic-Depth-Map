import Scene.Camera.Camera;
import VAOData;
import Scene.Intersection;
import Scene.Shading;
import Rendering.Materials.TexLODHelpers;
import Rendering.Materials.TexLODTypes;
import Utils.Math.PackedFormats;
//#include "Scene/Material/MaterialDefines.slangh"

#include "../StochasticDepthMapRT/Jitter.slangh"

// single depth texture
#define DEPTH_MODE_SINGLE 0
// two depth textures
#define DEPTH_MODE_DUAL 1
// single or dual depth texture + stochastic depth texture
#define DEPTH_MODE_STOCHASTIC 2
// raytraced
#define DEPTH_MODE_RAYTRACING 3
#define DEPTH_MODE_MACHINE_CLASSIFY 4
#define DEPTH_MODE_MACHINE_PREDICT 5
#define DEPTH_MODE_PERFECT_CLASSIFY 6
// use mipmaps to estimate depth, when mip0 is ambiguous
#define DEPTH_MODE_MIPMAPS 7 

#define sd_map_t float2

#define STOCHASTIC_DEPTH_RASTER 0
#define STOCHASTIC_DEPTH_RAY 1
#if STOCHASTIC_DEPTH_IMPL == STOCHASTIC_DEPTH_RASTER
#define STOCHASTIC_DEPTH_MAP Texture2DMS<sd_map_t>
#define LOAD_STOCHASTIC_SAMPLE(tex, xy, i) tex.Load(xy, i)
#else
#define STOCHASTIC_DEPTH_MAP Texture2DArray<sd_map_t>
#define LOAD_STOCHASTIC_SAMPLE(tex, xy, i) tex.Load(int4(xy, i, /*mipmap*/0))
#endif

// area where the halo effect remains constant at 0.0
#define CONST_RADIUS ((1.0 + gData.thickness) * data.radius - sphereStart)
#define HALO_RADIUS sphereStart
#define COMBINE_VIS(a,b) min(a,b)

#if DUAL_AO == 0
#define ao_t float
#define darkmap(x)
#else
#define ao_t float2
#define darkmap(x) x
#endif

#define NUM_DIRECTIONS 8
// normalized radius for each of the NUM_DIRECTION samples (distributed with radical inverse => see SSAO::setKernel() radius)
//static const float sampleRadius[NUM_DIRECTIONS] = { 0.608308673, 0.776627183, 0.417753726, 0.866025388, 0.518647850, 0.692805171, 0.291845083, 0.917883337 };
static const float sampleRadius[NUM_DIRECTIONS] = { 0.917883, 0.564429, 0.734504, 0.359545, 0.820004, 0.470149, 0.650919, 0.205215 };

cbuffer StaticCB
{
    VAOData gData;
}

cbuffer PerFrameCB
{
    float4x4 invViewMat;
    Camera gCamera;
    uint guardBand;
    bool useDepthLod;
    uint frameIndex;
}

SamplerState gNoiseSampler;
SamplerState gTextureSampler;

#ifndef DEPTH_MIPS
#define DEPTH_MIPS 1
#endif

Texture2D<float> gDepthTexMips[DEPTH_MIPS];
#define gDepthTex gDepthTexMips[0]
Texture2D<float> gDepthTex2;

// additional depth textures
STOCHASTIC_DEPTH_MAP gsDepthTex;

Texture2D<float4> gColor;

Texture2D<uint> gNormalTex;
Texture2D<float> gNoiseTex;

float3 loadNormal(float2 texC)
{
    uint packedNormal = gNormalTex[texC * gData.resolution];
    //return decodeNormal2x16(packedNormal);
    return decodeNormal2x8(packedNormal);
}

float2 getScreenClampedUV(float2 uvstart, float2 uvend)
{
    return saturate(uvend); // this actually does not make much of a difference but costs a little bit more...
}

float2 getSnappedUV(float2 uv, float2 resolution)
{
    float2 pixelCoord = floor(uv * resolution);
    return float2((pixelCoord.x + 0.5f) / resolution.x, (pixelCoord.y + 0.5f) / resolution.y);
}

float2 getSnappedUV(float2 uv)
{
    return getSnappedUV(uv, gData.resolution);
}



bool isSamePixel(float2 uv1, float2 uv2)
{
    //return false;
    return all(abs(uv1 - uv2) < gData.invResolution * 0.9);
    //return all(abs(uv1 - uv2) < gData.invResolution * 1.1); // this also ignores 1-pixel differences
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

int2 UVToPixel(float2 uv)
{
    float width, height;
    gDepthTex.GetDimensions(width, height);
    return int2(floor(uv * float2(width, height)));
}

float makeNonZero(float value, float epsilon)
{
    float absValue = max(abs(value), epsilon);
    return value >= 0 ? absValue : -absValue;
}

// get rid of shadowing around edges
// introduce a linear falloff function that starts with 0.0 when the sample depth intersects the front sphere exactly,
// and falls of to 1.0 when it gets further away from the sphere but closer to the camera.
// this also includes the constant radius, where visibility remains 0
float calcHaloVisibility(float objectSpaceZ, float sphereStart, float sphereEnd, float pdf, float radius)
{
    return saturate((objectSpaceZ - (1.0 + gData.thickness) * radius) / HALO_RADIUS)
        * (sphereStart - sphereEnd) / pdf; // this adjust the visibility to the sampling (hemi-)sphere
}

float calcSphereVisibility(float objectSpaceZ, float sphereStart, float sphereEnd, float pdf)
{
    float sampleRange = max(sphereStart - max(sphereEnd, objectSpaceZ), 0.0);
    return sampleRange / pdf;
}

float calcVisibility(float objectSpaceZ, float sphereStart, float sphereEnd, float pdf, float radius)
{
    return calcSphereVisibility(objectSpaceZ, sphereStart, sphereEnd, pdf)
         + calcHaloVisibility(objectSpaceZ, sphereStart, sphereEnd, pdf, radius);
}

float2 calcGatherUV(float2 uv, Texture2D<float> tex)
{
    float2 resolution;
    tex.GetDimensions(resolution.x, resolution.y);

    return (floor(uv * resolution - 0.5) + 1.0) / resolution;
}

float4 getTexelWeights(float2 uv, Texture2D<float> tex)
{
    float2 resolution;
    tex.GetDimensions(resolution.x, resolution.y);
    
    // calculate bilinear interpolation weights from uv coordinate
    float2 f = frac(uv * resolution - 0.5);
    // (-,+),(+,+),(+,-),(-,-)
    float4 w = float4((1.0 - f.x) * f.y, f.x * f.y, f.x * (1.0 - f.y), (1.0 - f.x) * (1.0 - f.y));
    return w;
}

float4 getTexelPointWeight(float4 w)
{
    int maxIdx = 0;
    float maxVal = w[0];
    for (int i = 1; i < 4; ++i)
    {
        if (w[i] > maxVal)
        {
            maxIdx = i;
            maxVal = w[i];
        }
    }
    float4 res = 0.0;
    for (int i = 1; i < 4; ++i)
        if (i == maxIdx)
            res[i] = 1.0;

    return res;
}



float3 RayToViewSpace(RayDesc ray, float t)
{
    return mul(gCamera.data.viewMat, float4(ray.Origin + ray.Direction * t, 1.0f)).xyz;
}

// z: positive linear depth in view space
// r: radius in view/world space
float2 ViewSpaceRadiusToUVRadius(float z, float r)
{
    //const float2 imageScale = 0.5 * float2(gCamera.data.frameWidth / gCamera.data.focalLength, gCamera.data.frameHeight / gCamera.data.focalLength);
    //float2 ndc = float2(r) / (imageScale * z); // radius in normalized device coordinates
    //return ndc * 0.5; // scale to uv radius
    return float2(r * gCamera.data.focalLength) / (float2(gCamera.data.frameWidth, gCamera.data.frameHeight) * z); // radius in normalized device coordinates
}

float GetAORadiusInPixels(float ViewDepth)
{
    // convert radius to screen pixels
    float2 radiusUV = ViewSpaceRadiusToUVRadius(ViewDepth, gData.radius);
    // convert uv radius to pixel radius
    return lerp(radiusUV.x * gData.resolution.x, radiusUV.y * gData.resolution.y, 0.5); // take mean between width and height radii TODO  test
}

float calcSampleImportance(
    float sampleRadius, float sampleLinearZ,
    //float3 sampleNormal, float3 centerNormal,
    float sphereStart, float sphereEnd, float objectSpaceZ,
    float3 sampleColor)
{
    float screen_radius = dot(ViewSpaceRadiusToUVRadius(sampleLinearZ, sampleRadius), float2(0.5));
    // clamp screen radius
    screen_radius = saturate(screen_radius / 20.0);
    
    //float normal_diff = abs(dot(sampleNormal, centerNormal));
    float dist = max(objectSpaceZ - sphereEnd, 0.0) / gData.radius;
    // clamp dist
    dist = saturate(dist / 5.0);
    
    float contrib = saturate((sphereStart - sphereEnd) / (2.0 * sphereStart));
    float luminance = saturate(dot(sampleColor, float3(0.2126, 0.7152, 0.0722)));
    luminance = pow(luminance, 1.0 / 2.4);
    
    return contrib * luminance * 0.999 * screen_radius * dist;
    //return 0.999 * screen_radius * dist;
}

struct BasicAOData
{
    float3 posV;
    float posVLength;
    float3 normal; // view space: -posV
    float3 tangent; // view space
    float3 bitangent; // view space
    float3 normalO; // sampling space (surface normal)
    float3 normalV; // view space (surface normal)
    
    float radiusInPixels;
    float radius; // world space radius. Usually gData.radius, but can be smaller if the screen space radius would be too large
    float _rayRasterBlend;
    //float linearDepth;

    // returns false if sample needs no shading (background)
    [mutating] bool Init(float2 texC)
    {
        float linearDepth = gDepthTex.SampleLevel(gTextureSampler, texC, 0);
        radiusInPixels = GetAORadiusInPixels(linearDepth);
        radius = gData.radius;
        // limit the pixel radius to maxPixelRadius to prevent samples from being distributed over the entire screen (bad for cache)
        if (radiusInPixels > gData.ssMaxRadius)
        {
            radius = radius / radiusInPixels * gData.ssMaxRadius;
            radiusInPixels = gData.ssMaxRadius;
        }
        _rayRasterBlend = saturate((radiusInPixels - gData.ssRadiusFadeEnd) / gData.ssRadiusFadeSize);
        
        if (radiusInPixels < 0.5)
            return false;

        posV = UVToViewSpace(texC, linearDepth);
        posVLength = length(posV);

        // view space normal of current pixel
        normalV = loadNormal(texC);
        if (dot(posV, normalV) > 0.0)
            normalV = -normalV;

        // Calculate tangent space (use random direction for tangent orientation)
        float randRotation = gNoiseTex.SampleLevel(gNoiseSampler, texC * gData.noiseScale, 0) * 2.0 * 3.141;
        float2 randDir = float2(sin(randRotation), cos(randRotation));
        //randDir = float2(1.0f, 0.0f);
    
        // determine tangent space
        normal = -posV / posVLength;
        bitangent = normalize(cross(normal, float3(randDir, 0.0f)));
        tangent = cross(bitangent, normal);

        // transfer view space normal to normal in object coordinates of the sampling sphere
        normalO = float3(dot(normalV, tangent), dot(normalV, bitangent), dot(normalV, normal));
        
        
        return true;
    }

    bool denyRays()
    {
        return radiusInPixels <= gData.ssRadiusFadeEnd;
    }

    // will force denyRays() to return false
    [mutating] void forceDenyRays()
    {
        radiusInPixels = 0.0;
    }
};

struct SampleAOData
{
    float sphereStart;
    float sphereEnd;
    float pdf; // 2 * sphereStart
    bool isInScreen;

    float2 samplePosUV; // possible out-of-screen sample location
    float2 rasterSamplePosUV; // raster clamped uv

    float visibility;
    float objectSpaceZ;

    float initialSamplePosLength;
    float radius; // world space radius of sample
    float3 initialSamplePosV;
    
    float screenSpaceRadius;
    float rayRasterBlend; // sample ray raster blend
    
    // returns false if the sample is invalid (below hemisphere)
    [mutating] bool Init(float2 texC, BasicAOData data, uint i)
    {
        // random angle on view space disc
        float alpha = (float(i) / NUM_DIRECTIONS) * 2.0 * 3.141;
        radius = sampleRadius[i] * data.radius; // radius on sampling unit sphere * world space radius
        float2 dir = radius * float2(sin(alpha), cos(alpha)); // world space direction

        const float sphereHeight = sqrt(data.radius * data.radius - radius * radius);
        pdf = 2.0 * sphereHeight;
        
        // determine distance within [-sphereHeight, +sphereHeight]
        sphereStart = sphereHeight; // in object coordinates (bigger is closer to the camera)
        sphereEnd = -sphereHeight; // in object coordinates (smaller is futher from the camera)

        { // HEMISPHERE SAMPLING
            //float zIntersect = -dot(rand.xy, normalO.xy) / normalO.z;
            float zIntersect = -dot(dir.xy, data.normalO.xy) / makeNonZero(data.normalO.z, 0.0001);
            float zIntersectClamped = clamp(zIntersect, -sphereHeight, sphereHeight);
            sphereEnd = zIntersectClamped;
        }

        if ((sphereStart - sphereEnd) / pdf <= 0.1)
        {
            return false; // skip sample (no visibility)
        }
        
        // sample position calculate uv position of sample
        initialSamplePosV = data.posV + data.tangent * dir.x + data.bitangent * dir.y;
        initialSamplePosLength = length(initialSamplePosV);
        samplePosUV = ViewSpaceToUV(initialSamplePosV);
        visibility = 0.0;
        objectSpaceZ = 0.0;
        screenSpaceRadius = length(float2((texC - samplePosUV) * gData.resolution));
        rayRasterBlend = saturate((screenSpaceRadius - gData.ssRadiusFadeEnd) / gData.ssRadiusFadeSize);
        //rayRasterBlend = data._rayRasterBlend;
        
        float2 screenUv = getScreenClampedUV(texC, samplePosUV); // clip to screen border
        isInScreen = all(samplePosUV == screenUv);
            
        rasterSamplePosUV = screenUv;
        rasterSamplePosUV = getSnappedUV(rasterSamplePosUV); // snap to pixel center
        
        return true;
    }

    float rectifyObjectSpaceZ(BasicAOData data, float objectSpaceZ, float3 samplePosV)
    {
        float3 projSamplePosV = samplePosV - objectSpaceZ * data.normal; // project to sampling plane
        float originalSampleDist = radius; // == sampleRadius[i] * data.radius
        float rasterSampleDist = distance(data.posV, projSamplePosV);
        return objectSpaceZ * originalSampleDist / rasterSampleDist;
    }

    float calcObjectSpaceZFromPlane(BasicAOData data, float3 samplePosV, float3 sampleNormalV)
    {
        // ray-plane intersection:
        // plane: samplePosV, sampleNormalV
        // ray: initialSamplePosV, data.normal
        float denom = dot(sampleNormalV, data.normal);
        if (abs(denom) < 1e-6) // almost perpendicular
            return dot(samplePosV - data.posV, data.normal); // default computation

        return dot(samplePosV - initialSamplePosV, sampleNormalV) / denom;
    }
    
    // raster impl.
    float calcObjectSpaceZ(BasicAOData data, float2 uv, Texture2D<float> depthTex, uint lod = 0u)
    {
        float linearSampleDepth;
        if (lod == 0u)
            linearSampleDepth = depthTex.SampleLevel(gTextureSampler, uv, 0.0);
        else // useGather
        {
            float2 gatherUV = calcGatherUV(uv, depthTex);
            float4 d = depthTex.Gather(gTextureSampler, gatherUV);
            float4 w = getTexelWeights(uv, depthTex);

        // closest depth
            float dRef = dot(getTexelPointWeight(w), d);
        //float dRef = max(max(d[0], d[1]), max(d[2], d[3]));

            float lowT = pow(0.9, lod);
            float highT = pow(1.1, lod);
            for (uint i = 0; i < 4; ++i)
            {
                if (d[i] < lowT * dRef || d[i] > highT * dRef)
                    w[i] = 0.0; // not similar enough => reject
            }
            w = w / (w[0] + w[1] + w[2] + w[3]);

            linearSampleDepth = dot(w, d);
        //linearSampleDepth = dRef;
        //linearSampleDepth = depthTex.SampleLevel(gTextureSampler, uv, 0.0);
        }
    
        float3 samplePosV = UVToViewSpace(uv, linearSampleDepth);
        // adjust view space position
        //samplePosV = initialSamplePosV + dot(data.normalV, samplePosV - initialSamplePosLength) * data.normalV;
        
        // the object is the sphere centered at posV with the above tangent space (positive values are closer to the camera)
        float objectSpaceZ = dot(samplePosV - data.posV, data.normal);
        // correct the objectSpaceZPos
        //objectSpaceZ = rectifyObjectSpaceZ(data, objectSpaceZ, samplePosV);
        // this is usually too expensive (x1.5) and only distant samples benefit from it
        //float3 sampleNormalV = loadNormal(uv); //gNormalTex.SampleLevel(gTextureSampler, uv, 0.0).xyz);
        //objectSpaceZ = calcObjectSpaceZFromPlane(data, samplePosV, sampleNormalV);
        
        return objectSpaceZ;
    }

    // ray impl.
    float calcObjectSpaceZ(float3 posV, float3 normal, RayDesc ray, float t)
    {
        float3 samplePosV = RayToViewSpace(ray, t);
        float objectSpaceZ = dot(samplePosV - posV, normal);
        return objectSpaceZ;
    }

    bool requireRay(BasicAOData data)
    {
        return objectSpaceZ > sphereStart + CONST_RADIUS && rayRasterBlend > 0.0;
    }

    [mutating] void evalPrimaryVisibility(BasicAOData data)
    {
        uint lod = 0;
        if (useDepthLod)
            lod = (uint)max(int(log2(screenSpaceRadius / 8.0) + 0.5), 0);
        objectSpaceZ = calcObjectSpaceZ(data, rasterSamplePosUV, gDepthTexMips[lod], lod);
        visibility = calcVisibility(objectSpaceZ, sphereStart, sphereEnd, pdf, data.radius);

        if (PRIMARY_DEPTH_MODE == DEPTH_MODE_MIPMAPS)
        {
            lod += 1;
            for (; lod < uint(DEPTH_MIPS); ++lod)
            {
                // take min object space Z
                objectSpaceZ = min(objectSpaceZ, calcObjectSpaceZ(data, rasterSamplePosUV, gDepthTexMips[lod], lod));
                visibility = COMBINE_VIS(visibility, calcVisibility(objectSpaceZ, sphereStart, sphereEnd, pdf, data.radius));
                if (!requireRay(data))
                    break;
            }

        }
    }

    [mutating] void evalDualVisibility(BasicAOData data)
    {
        if (!requireRay(data))
            return; // not needed
        
        objectSpaceZ = min(objectSpaceZ, calcObjectSpaceZ(data, rasterSamplePosUV, gDepthTex2));
        visibility = COMBINE_VIS(visibility, calcVisibility(objectSpaceZ, sphereStart, sphereEnd, pdf, data.radius));
    }
};

struct RayData // cannot be compressed to half floats => no diff in rendering time + insufficient visual quality
{
    float tLastFrontFaceHalo; // ray min
    float tFirstFrontFaceInside; // ray max
    float tConstRadiusStart;
    float tSphereStart;
};

void traceAORay(RayDesc ray, inout RayData rayData);

ao_t calcAO2(uint2 svPos, uint mask)
{
    float2 texC = (float2(svPos) + 0.5) * gData.invResolution;

    BasicAOData data;
    data.Init(texC);

    ao_t visibility = 0.0;

    uint i = 0;
    //[loop] while(mask != 0u)
    [unroll]
    for (uint j = 0; j < NUM_DIRECTIONS; j++)
    {
        if (mask == 0u)
            break; // no bits set anymore

        // modify loop to only go through the set bits in mask
        //[loop] while ((mask & 1u) == 0u)
        //for (uint k = 0; k < (NUM_DIRECTIONS - j) && (mask & 1u) == 0u; k++) // this is too complicated for current compiler..  
        [unroll]
        for (uint k = 0; k < NUM_DIRECTIONS && k < NUM_DIRECTIONS - j && (mask & 1u) == 0u; k++) // first condition is for unrolling, second is for better unrolling
        {
            // shift mask an increase i
            mask = mask >> 1;
            ++i;
        }

        SampleAOData s;
        s.Init(texC, data, i);

        // subtract old visibility from raster (will be replaced with new visibility)
        if (PRIMARY_DEPTH_MODE != DEPTH_MODE_DUAL) // DEPTH_MODE = SINGLE (or classify)
            s.evalPrimaryVisibility(data);
        else // DEPTH_MODE == DUAL
            s.evalDualVisibility(data);
        float curVisibility = s.visibility;
        visibility.x -= s.visibility; // subtract old visibility from raster for bright channel
        
        if (SECONDARY_DEPTH_MODE == DEPTH_MODE_STOCHASTIC)
        {
            float sdwidth, sdheight, elements;
            gsDepthTex.GetDimensions(sdwidth, sdheight, elements);

            int2 pixelCoord = int2(floor(s.rasterSamplePosUV * float2(sdwidth, sdheight)));
            //float2 sdSampleUV = getSnappedUV(s.rasterSamplePosUV, float2(sdwidth, sdheight));
            float2 sdSampleUV = (pixelCoord + randomJitter(pixelCoord)) / float2(sdwidth, sdheight);
            const float depthRange = gCamera.data.farZ - gCamera.data.nearZ;
            const float depthOffset = gCamera.data.nearZ;
            bool stochInside = false;
            [unroll]
            for (uint i = 0; i < MSAA_SAMPLES; ++i)
            {
                float2 sd_data = LOAD_STOCHASTIC_SAMPLE(gsDepthTex, pixelCoord, i);
                float linearSampleDepth = sd_data.x;
                // linearSampleDepth is in [0, 1] => scale accordingly
                linearSampleDepth = linearSampleDepth * depthRange + depthOffset;
                float3 samplePosV = UVToViewSpace(sdSampleUV, linearSampleDepth);
                float objectSpaceZ = dot(samplePosV - data.posV, data.normal);
                //objectSpaceZ = s.rectifyObjectSpaceZ(data, objectSpaceZ, samplePosV);
                #if STOCH_MAP_NORMALS
                float3 sd_normal = decodeNormal2x16(asuint(sd_data.y));
                objectSpaceZ = s.calcObjectSpaceZFromPlane(data, samplePosV, sd_normal);
                #endif
                
                //if (objectSpaceZ <= data.radius + gData.thickness * data.radius && objectSpaceZ >= -data.radius)
                //    stochInside = true;
                float newVisibility = calcVisibility(objectSpaceZ, s.sphereStart, s.sphereEnd, s.pdf, data.radius);
                curVisibility = COMBINE_VIS(curVisibility, newVisibility);
            }
            curVisibility = lerp(s.visibility, curVisibility, s.rayRasterBlend);
            // if (!stochInside) visibility.y -= curVisibility; // keep dark term dark (+= curVisibility will be performed at the end)

        }
        else if (SECONDARY_DEPTH_MODE == DEPTH_MODE_RAYTRACING)
        {
            // to be consistent with the rasterizer, we snap the uv coordinate as well to the pixel center,
            // but we do not clip it since we can shoot outside of the screen space
            //samplePosUV = getScreenClampedUV(texC, samplePosUV);
            s.samplePosUV = getSnappedUV(s.samplePosUV); // snap to pixel center
            
            float3 sampleDirV = normalize(UVToViewSpace(s.samplePosUV, 1.0)); // get sample direction in view space
            
            RayDesc ray;
            ray.Origin = gCamera.data.posW; // offset a little bit in normal direction
            ray.Direction = mul(float3x3(invViewMat), sampleDirV);

            // ray query or ray pipeline implementation
            RayData rayData;
            rayData.tLastFrontFaceHalo = (data.posVLength - s.sphereStart - data.radius - gData.thickness * data.radius) * s.initialSamplePosLength / data.posVLength; // min (haloStart)
            rayData.tFirstFrontFaceInside = (data.posVLength - s.sphereEnd) * s.initialSamplePosLength / data.posVLength; // max (sphereEnd)
            rayData.tConstRadiusStart = (data.posVLength - data.radius - gData.thickness * data.radius) * s.initialSamplePosLength / data.posVLength;
            rayData.tSphereStart = (data.posVLength - s.sphereStart) * s.initialSamplePosLength / data.posVLength;

            ray.TMin = max(rayData.tLastFrontFaceHalo, 0.0);
            ray.TMax = rayData.tFirstFrontFaceInside; // sphereEnd

            const float epsilon = data.radius * 0.01;
            // include the value of the depth buffer when choosing TMin to save some traversal time
            if (s.isInScreen)
                ray.TMin = max(ray.TMin, (data.posVLength - s.objectSpaceZ) * s.initialSamplePosLength / data.posVLength + epsilon);
            else
                curVisibility = 1.0;
            
            traceAORay(ray, rayData);

            float sphereVisibility = calcVisibility(data.posVLength - rayData.tFirstFrontFaceInside * data.posVLength / s.initialSamplePosLength, s.sphereStart, s.sphereEnd, s.pdf, data.radius);
            float haloVisibility = calcHaloVisibility(data.posVLength - rayData.tLastFrontFaceHalo * data.posVLength / s.initialSamplePosLength, s.sphereStart, s.sphereEnd, s.pdf, data.radius);

            float rayVisibility = min(curVisibility, min(sphereVisibility, haloVisibility));
            curVisibility = lerp(s.visibility, rayVisibility, s.rayRasterBlend);
        }

        visibility += curVisibility;
        
        // advance mask for next iteration
        mask = mask >> 1;
        ++i;
    }
    
    visibility *= 2.0 / float(NUM_DIRECTIONS);
    return visibility;
}

#define AO_HIT_IGNORE 0
#define AO_HIT_ACCEPT 1
#define AO_HIT_ACCEPT_AND_END 2

ExplicitRayConesLodTextureSampler computeLod(VertexData v, float3 rayDir, float t)
{
    RayCone rc = RayCone(0.0, RAY_CONE_SPREAD);
    rc = rc.propagateDistance(t);
    float lambda = rc.computeLOD(v.coneTexLODValue, rayDir, v.faceNormalW);
    return ExplicitRayConesLodTextureSampler(lambda);
}

// returns any of the above A0_HIT defines
uint aoAnyHit(inout RayData rayData, float t, const TriangleHit hit, bool frontFace, float3 rayDir)
{
    const uint materialID = gScene.getMaterialID(hit.instanceID);
    const MaterialHeader header = gScene.materials.materialData[materialID].header;

    bool isAlphaTested = header.getAlphaMode() == AlphaMode::Mask;
#if USE_ALPHA_TEST
    if (isAlphaTested)
    {
        const VertexData v = gScene.getVertexData(hit);
        //if (gScene.materials.alphaTest(v, materialID, computeLod(v, rayDir, t)))
        if (gScene.materials.alphaTest(v, materialID, 0.0)) // no lods
            return AO_HIT_IGNORE; // alpha test failed => ignore this triangle
    }
#endif

    frontFace = frontFace || header.isDoubleSided() || isAlphaTested;
    if (!frontFace)
        return AO_HIT_IGNORE; // this is just for rasterizer compability

    if (t <= rayData.tSphereStart)
    {
        rayData.tLastFrontFaceHalo = max(rayData.tLastFrontFaceHalo, t);
        if (t >= rayData.tConstRadiusStart)
            return AO_HIT_ACCEPT_AND_END; // we can stop the query, because this will set the visibility to zero
    }
    else // inside sphere
    {
        rayData.tFirstFrontFaceInside = min(rayData.tFirstFrontFaceInside, t);
        return AO_HIT_ACCEPT; // since we save the min, we can commit TMax here
    }

    return AO_HIT_IGNORE;
}
