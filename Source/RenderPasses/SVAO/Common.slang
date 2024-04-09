import Scene.Camera.Camera;
import VAOData;
import Scene.Intersection;
import Scene.Shading;
import Rendering.Materials.TexLODHelpers;
import Rendering.Materials.TexLODTypes;
import Utils.Math.PackedFormats;
//#include "Scene/Material/MaterialDefines.slangh"

#include "../StochasticDepthMapRT/Jitter.slangh"

#define AO_KERNEL_VAO 0
#define AO_KERNEL_HBAO 1

// single depth texture
#define DEPTH_MODE_SINGLE 0
// two depth textures
#define DEPTH_MODE_DUAL 1
// single or dual depth texture + stochastic depth texture
#define DEPTH_MODE_STOCHASTIC 2
// raytraced
#define DEPTH_MODE_RAYTRACING 3

#define sd_map_t float2

#define STOCHASTIC_DEPTH_RASTER 0
#define STOCHASTIC_DEPTH_RAY 1
#if STOCHASTIC_DEPTH_IMPL == STOCHASTIC_DEPTH_RASTER
#define STOCHASTIC_DEPTH_MAP Texture2DMS<sd_map_t>
#define LOAD_STOCHASTIC_SAMPLE(tex, xy, i) tex.Load(xy, i)
#else
#define STOCHASTIC_DEPTH_MAP Texture2D<float4>
//#define LOAD_STOCHASTIC_SAMPLE(tex, xy, i) tex.Load(int4(xy, i, /*mipmap*/0))
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

// normalized radius for each of the NUM_DIRECTION samples (distributed with radical inverse => see SSAO::setKernel() radius)
//static const float sampleRadius[NUM_DIRECTIONS] = { 0.608308673, 0.776627183, 0.417753726, 0.866025388, 0.518647850, 0.692805171, 0.291845083, 0.917883337 };
#if AO_KERNEL == AO_KERNEL_VAO
#if NUM_DIRECTIONS == 8
static const float sampleRadius[8] = { 0.917883, 0.564429, 0.734504, 0.359545, 0.820004, 0.470149, 0.650919, 0.205215 };
#elif NUM_DIRECTIONS == 16
static const float sampleRadius[16] = {0.949098221604059, 0.5865639019441775, 0.7554681720909893, 0.3895439574863043, 0.8425560503012255, 0.4948003867747738, 0.6719196866381647, 0.25203100417434543, 0.8908588816103737, 0.5418210823278604, 0.7136427497994143, 0.32724136087586453, 0.7980920320691521, 0.4445340224611676, 0.6297373536812639, 0.1447182620692375};
#else // 32
static const float sampleRadius[32] = {0.9682458365518543, 0.5974803093982587, 0.7660169295429302, 0.4038472576817624, 0.8541535023444914, 0.5068159098187986, 0.6823727109604635, 0.2726076670970059, 0.904018191941786, 0.5531894754180758, 0.7240656647095169, 0.34372202910162664, 0.8089818132350507, 0.45747336127867605, 0.640354849019649, 0.17748061996818404, 0.9327350969376332, 0.5755500192397054, 0.7449678114312224, 0.37479566486456295, 0.8311856199411515, 0.4825843210309559, 0.6614378277661477, 0.22975243551455923, 0.878233108646881, 0.5303115209931901, 0.7032256306171377, 0.3099952198410562, 0.7873133907642258, 0.43130429537268, 0.6190581352335289, 0.10219580968897692};
#endif
#endif
#if AO_KERNEL == AO_KERNEL_HBAO
#if NUM_DIRECTIONS == 8
static const float sampleRadius[8] = {0.019897607325877215, 0.3239192018939078, 0.15013283288204182, 0.5608856339193332, 0.07874804859295396, 0.4306374970658152, 0.23159241868180838, 0.74770696488701};
#elif NUM_DIRECTIONS == 16
static const float sampleRadius[16] = {0.008364792005390745, 0.29968419137477154, 0.13131974798930376, 0.5251597224509892, 0.06264063727314514, 0.40226410430222115, 0.21027995621089465, 0.6906178807859765, 0.03303993608633204, 0.34903099295095424, 0.16956281924775551, 0.5996160679614535, 0.09559795810145842, 0.46040865279052423, 0.25357218870257175, 0.8218290863578166};
#else // 32
static const float sampleRadius[32] = {0.0035168784979124203, 0.28787249889929795, 0.12214740408236834, 0.5082189968610005, 0.05489041689357717, 0.38854375322009427, 0.19986558164830323, 0.6656225173745592, 0.02630214826181389, 0.33636038195532914, 0.15977097044845298, 0.579825376399601, 0.08708424832212604, 0.44533522627083877, 0.24249692822679572, 0.7816464549941924, 0.013886447731081395, 0.3116969449839127, 0.14064876764650994, 0.5426920213922799, 0.07059703986067731, 0.41628837439340993, 0.22085459126773643, 0.7177502077720759, 0.04006955250785802, 0.36194276200351894, 0.17950859741413544, 0.6203897476558216, 0.10428292232859922, 0.47588885313824597, 0.2648228762567681, 0.8740952987729764};
#endif
#endif

cbuffer StaticCB
{
    VAOData gData;
}

cbuffer PerFrameCB
{
    float4x4 invViewMat;
    Camera gCamera;
    uint guardBand;
    uint frameIndex;
}

SamplerState gNoiseSampler;
SamplerState gTextureSampler;


Texture2D<float> gDepthTex;
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

float2 getSDClampedUV(float2 uvstart, float2 uvend)
{
    float2 offset = float(gData.sdGuard) / gData.lowResolution;
    return clamp(uvend, -offset, 1.0 + offset);
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

// converts from screen uv coordinates to pixel coordinates of stochastic depth map.
// UV can be negative or greater than one if the guard band for the SD-map is used (gData.sdGuard)
int2 UVToSDPixel(float2 uv)
{
    int2 pixel = int2(floor(uv * gData.lowResolution)) + int2(gData.sdGuard);
    return clamp(pixel, 0, int2(gData.lowResolution) + gData.sdGuard * 2 - 1);
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
    
        // determine tangent space
        normal = -posV / posVLength;
        bitangent = normalize(cross(normal, float3(randDir, 0.0f)));
        tangent = cross(bitangent, normal);

        // transfer view space normal to normal in object coordinates of the sampling sphere
        normalO = float3(dot(normalV, tangent), dot(normalV, bitangent), dot(normalV, normal));
        
        
        return true;
    }

    static float finalize(float avgAO)
    {
        if(AO_KERNEL == AO_KERNEL_HBAO) avgAO = saturate(1.0 - 2.0 * avgAO);
        return pow(avgAO, gData.exponent);
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
    
    // returns false if the sample is invalid (below hemisphere)
    [mutating] bool Init(float2 texC, BasicAOData data, uint i)
    {
        // random angle on view space disc
        float alpha = (float(i) / NUM_DIRECTIONS) * 2.0 * 3.141;
        radius = sampleRadius[i] * data.radius; // radius on sampling unit sphere * world space radius
        float2 dir = radius * float2(sin(alpha), cos(alpha)); // world space direction

        const float sphereHeight = sqrt(data.radius * data.radius - radius * radius);
        if(AO_KERNEL == AO_KERNEL_VAO)
            pdf = 2.0 * sphereHeight;
        else
            pdf = 0.9 * pow(1.0 - sampleRadius[i], 1.5);
        
        // determine distance within [-sphereHeight, +sphereHeight]
        sphereStart = sphereHeight; // in object coordinates (bigger is closer to the camera)
        sphereEnd = -sphereHeight; // in object coordinates (smaller is futher from the camera)

        { // HEMISPHERE SAMPLING
            //float zIntersect = -dot(rand.xy, normalO.xy) / normalO.z;
            float zIntersect = -dot(dir.xy, data.normalO.xy) / makeNonZero(data.normalO.z, 0.0001);
            float zIntersectClamped = clamp(zIntersect, -sphereHeight, sphereHeight);
            sphereEnd = zIntersectClamped;
        }

        if ((sphereStart - sphereEnd) / (2.0 * sphereHeight) <= 0.1)
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
        // screenSpaceRadius < 1 handled in first ao pass (isSamplePixel)

        float2 screenUv = getScreenClampedUV(texC, samplePosUV); // clip to screen border
        isInScreen = all(samplePosUV == screenUv);
            
        rasterSamplePosUV = screenUv;
        rasterSamplePosUV = getSnappedUV(rasterSamplePosUV); // snap to pixel center
        
        return true;
    }

    //float rectifyObjectSpaceZ(BasicAOData data, float objectSpaceZ, float3 samplePosV)
    //{
    //    float3 projSamplePosV = samplePosV - objectSpaceZ * data.normal; // project to sampling plane
    //    float originalSampleDist = radius; // == sampleRadius[i] * data.radius
    //    float rasterSampleDist = distance(data.posV, projSamplePosV);
    //    return objectSpaceZ * originalSampleDist / rasterSampleDist;
    //}

    //float calcObjectSpaceZFromPlane(BasicAOData data, float3 samplePosV, float3 sampleNormalV)
    //{
    //    // ray-plane intersection:
    //    // plane: samplePosV, sampleNormalV
    //    // ray: initialSamplePosV, data.normal
    //    float denom = dot(sampleNormalV, data.normal);
    //    if (abs(denom) < 1e-6) // almost perpendicular
    //        return dot(samplePosV - data.posV, data.normal); // default computation
    //
    //    return dot(samplePosV - initialSamplePosV, sampleNormalV) / denom;
    //}
    
    float HBAOKernel(BasicAOData data, float3 S)
    {
        float3 P = data.posV;
        float3 V = S - P;
        float NdotVBias = 0.1;

        float angleTerm = saturate(dot(data.normalV, normalize(V)) - NdotVBias); // 1 for directly above
        float distanceTerm = saturate(1.0 - dot(V, V) / (gData.radius * gData.radius)); // falloff: 0 for far away. 1 for nearby
        return angleTerm * distanceTerm;
    }

    float3 calcSamplePosV(BasicAOData data, float2 uv, Texture2D<float> depthTex)
    {
        float linearSampleDepth = depthTex.SampleLevel(gTextureSampler, uv, 0.0);
    
        float3 samplePosV = UVToViewSpace(uv, linearSampleDepth);
        return samplePosV;
    }
    
    float calcObjectSpaceZ(BasicAOData data, float3 samplePosV)
    {
        float objectSpaceZ = dot(samplePosV - data.posV, data.normal);
        
        return objectSpaceZ;
    }

    // ray impl.
    //float calcObjectSpaceZ(float3 posV, float3 normal, RayDesc ray, float t)
    //{
    //    float3 samplePosV = RayToViewSpace(ray, t);
    //    float objectSpaceZ = dot(samplePosV - posV, normal);
    //    return objectSpaceZ;
    //}

    bool requireRay(BasicAOData data)
    {
        if(AO_KERNEL == AO_KERNEL_VAO)
            return objectSpaceZ > sphereStart + CONST_RADIUS && screenSpaceRadius > gData.ssRadiusCutoff;
        // HBAO
        return objectSpaceZ > max(sphereStart, data.radius * 0.1) && screenSpaceRadius > gData.ssRadiusCutoff;
    }

    [mutating] void addSample(BasicAOData data, float3 samplePosV, bool init = false)
    {
        // update object space z
        float oz = calcObjectSpaceZ(data, samplePosV);
        if(init) objectSpaceZ = oz;
        else objectSpaceZ = min(objectSpaceZ, oz);

        // update visibility
        if(AO_KERNEL == AO_KERNEL_VAO)
        {
            float v = calcVisibility(oz, sphereStart, sphereEnd, pdf, data.radius);
            if(init) visibility = v;
            else visibility = min(visibility, v);
        }
        if(AO_KERNEL == AO_KERNEL_HBAO)
        {
            float v = saturate(HBAOKernel(data, samplePosV) / pdf);
            if(init) visibility = v;
            else visibility = max(visibility, v);
        }
    }

    [mutating] void resetSample()
    {
        if(AO_KERNEL == AO_KERNEL_VAO) visibility = 1.0;
        if(AO_KERNEL == AO_KERNEL_HBAO) visibility = 0.0;
        objectSpaceZ = 3.402823466e+38F;
    }

    [mutating] void evalPrimaryVisibility(BasicAOData data)
    {
        float3 samplePosV = calcSamplePosV(data, rasterSamplePosUV, gDepthTex);
        addSample(data, samplePosV, true);
    }

    [mutating] void evalDualVisibility(BasicAOData data, bool init = false)
    {
        if (!requireRay(data))
            return; // not needed
        
        float3 samplePosV = calcSamplePosV(data, rasterSamplePosUV, gDepthTex2);
        addSample(data, samplePosV, init);
    }
};

struct RayData // cannot be compressed to half floats => no diff in rendering time + insufficient visual quality
{
#if AO_KERNEL == AO_KERNEL_VAO
    float tLastFrontFaceHalo; // ray min
    float tFirstFrontFaceInside; // ray max
    float tConstRadiusStart;
    float tSphereStart;
#endif
#if AO_KERNEL == AO_KERNEL_HBAO
    float tFirst; // ray t of first hit inside sphere
#endif
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
            s.evalDualVisibility(data, true); // force init
        
        visibility.x -= s.visibility; // subtract old visibility from raster for bright channel
        
        if (SECONDARY_DEPTH_MODE == DEPTH_MODE_STOCHASTIC)
        {            
            //int2 pixelCoord = UVToSDPixel(s.rasterSamplePosUV);
            int2 pixelCoord = UVToSDPixel(s.samplePosUV);
            
            //float2 sdSampleUV = getSnappedUV(s.rasterSamplePosUV, float2(sdwidth, sdheight));
            // calc uv space position based on primary frame buffer size
            float2 sdSampleUV = (pixelCoord - gData.sdGuard + randomJitter(pixelCoord)) / gData.lowResolution;
            const float depthRange = gCamera.data.farZ - gCamera.data.nearZ;
            const float depthOffset = gCamera.data.nearZ;
            bool stochInside = false;
            #if STOCHASTIC_DEPTH_IMPL != STOCHASTIC_DEPTH_RASTER
            float4 sddepths = gsDepthTex[pixelCoord];
            #endif

            if (!s.isInScreen) s.resetSample();

            [unroll]
            for (uint i = 0; i < MSAA_SAMPLES; ++i)
            {
                #if STOCHASTIC_DEPTH_IMPL == STOCHASTIC_DEPTH_RASTER
                float2 sd_data = LOAD_STOCHASTIC_SAMPLE(gsDepthTex, pixelCoord, i);
                float linearSampleDepth = sd_data.x;
                #else
                float linearSampleDepth = sddepths[i];
                #endif
                // linearSampleDepth is in [0, 1] => scale accordingly
                linearSampleDepth = linearSampleDepth * depthRange + depthOffset;
                float3 samplePosV = UVToViewSpace(sdSampleUV, linearSampleDepth);
                s.addSample(data, samplePosV);
            }
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
            #if AO_KERNEL == AO_KERNEL_VAO
            rayData.tLastFrontFaceHalo = (data.posVLength - s.sphereStart - data.radius - gData.thickness * data.radius) * s.initialSamplePosLength / data.posVLength; // min (haloStart)
            rayData.tFirstFrontFaceInside = (data.posVLength - s.sphereEnd) * s.initialSamplePosLength / data.posVLength; // max (sphereEnd)
            rayData.tConstRadiusStart = (data.posVLength - data.radius - gData.thickness * data.radius) * s.initialSamplePosLength / data.posVLength;
            rayData.tSphereStart = (data.posVLength - s.sphereStart) * s.initialSamplePosLength / data.posVLength;

            ray.TMin = max(rayData.tLastFrontFaceHalo, 0.0);
            ray.TMax = rayData.tFirstFrontFaceInside; // sphereEnd

            #endif
            #if AO_KERNEL == AO_KERNEL_HBAO
            rayData.tFirst = 0.0;

            ray.TMin = (data.posVLength - s.sphereStart) * s.initialSamplePosLength / data.posVLength; // sphereStart
            ray.TMax = (data.posVLength - s.sphereEnd) * s.initialSamplePosLength / data.posVLength; // sphereEnd
            #endif

            if (!s.isInScreen) s.resetSample();

            const float epsilon = data.radius * 0.01;
            // include the value of the depth buffer when choosing TMin to save some traversal time
            if (s.isInScreen)
                ray.TMin = max(ray.TMin, (data.posVLength - s.objectSpaceZ) * s.initialSamplePosLength / data.posVLength + epsilon);
                
            
            traceAORay(ray, rayData);

            #if AO_KERNEL == AO_KERNEL_VAO
            float sphereVisibility = calcVisibility(data.posVLength - rayData.tFirstFrontFaceInside * data.posVLength / s.initialSamplePosLength, s.sphereStart, s.sphereEnd, s.pdf, data.radius);
            float haloVisibility = calcHaloVisibility(data.posVLength - rayData.tLastFrontFaceHalo * data.posVLength / s.initialSamplePosLength, s.sphereStart, s.sphereEnd, s.pdf, data.radius);

            s.visibility = min(s.visibility, min(sphereVisibility, haloVisibility));
            
            #else // HBAO
            float3 samplePosW = ray.Origin + ray.Direction * rayData.tFirst;
            float3 samplePosV = mul(gCamera.data.viewMat, float4(samplePosW, 1.0)).xyz;
            s.addSample(data, samplePosV);
            #endif
        }

        visibility += s.visibility;
        
        // advance mask for next iteration
        mask = mask >> 1;
        ++i;
    }
    
    visibility *= 1.0 / float(NUM_DIRECTIONS);
    if(AO_KERNEL == AO_KERNEL_VAO) visibility *= 2.0;
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

    #if AO_KERNEL == AO_KERNEL_VAO
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
    
    #endif
    #if AO_KERNEL == AO_KERNEL_HBAO
    return AO_HIT_ACCEPT; // since we save the min, we can commit TMax here
    #endif
}
