import Scene.Camera.Camera;
import VAOData;
import Scene.Intersection;
import Scene.Shading;

//#include "Scene/Material/MaterialDefines.slangh"

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

#define STOCHASTIC_DEPTH_RASTER 0
#define STOCHASTIC_DEPTH_RAY 1
#if STOCHASTIC_DEPTH_IMPL == STOCHASTIC_DEPTH_RASTER
#define STOCHASTIC_DEPTH_MAP Texture2DMS<float>
#define LOAD_STOCHASTIC_SAMPLE(tex, xy, i) tex.Load(xy, i)
#else
#define STOCHASTIC_DEPTH_MAP Texture2DArray<float>
#define LOAD_STOCHASTIC_SAMPLE(tex, xy, i) tex.Load(int4(xy, i, /*mipmap*/0))
#endif

// area where the halo effect remains constant at 0.0
#define CONST_RADIUS ((1.0 + gData.thickness) * gData.radius - sphereStart)
#define HALO_RADIUS sphereStart
#define COMBINE_VIS(a,b) min(a,b)

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
}

SamplerState gNoiseSampler;
SamplerState gTextureSampler;

Texture2D<float> gDepthTex;
Texture2D<float> gDepthTex2;

// additional depth textures
STOCHASTIC_DEPTH_MAP gsDepthTex;

Texture2D<float4> gColor;

Texture2D gNormalTex;
Texture2D<float> gNoiseTex;


#define MTL_DOUBLE_SIDED 1
Texture2D<uint> gMatDoubleSided; // double sided flag

float2 getScreenClampedUV(float2 uvstart, float2 uvend)
{
    return saturate(uvend); // this actually does not make much of a difference but costs a little bit more...
}

float2 getSnappedUV(float2 uv)
{
    float2 pixelCoord = floor(uv * gData.resolution);
    return float2((pixelCoord.x + 0.5f) / gData.resolution.x, (pixelCoord.y + 0.5f) / gData.resolution.y);
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
float calcHaloVisibility(float objectSpaceZ, float sphereStart, float sphereEnd, float pdf)
{
    return saturate((objectSpaceZ - sphereStart - CONST_RADIUS) / HALO_RADIUS)
        * (sphereStart - sphereEnd) / pdf; // this adjust the visibility to the sampling (hemi-)sphere
}

float calcSphereVisibility(float objectSpaceZ, float sphereStart, float sphereEnd, float pdf)
{
    float sampleRange = max(sphereStart - max(sphereEnd, objectSpaceZ), 0.0);
    return sampleRange / pdf;
}

float calcVisibility(float objectSpaceZ, float sphereStart, float sphereEnd, float pdf)
{
    return calcSphereVisibility(objectSpaceZ, sphereStart, sphereEnd, pdf)
         + calcHaloVisibility(objectSpaceZ, sphereStart, sphereEnd, pdf);
}

float calcObjectSpaceZ(float3 posV, float3 normal, float2 uv, Texture2D<float> depthTex)
{
    float linearSampleDepth = depthTex.SampleLevel(gTextureSampler, uv, 0);
    float3 samplePosV = UVToViewSpace(uv, linearSampleDepth);
            // the object is the sphere centered at posV with the above tangent space (positive values are closer to the camera)
    float objectSpaceZ = dot(samplePosV - posV, normal);
    return objectSpaceZ;
}

float3 RayToViewSpace(RayDesc ray, float t)
{
    return mul(gCamera.data.viewMat, float4(ray.Origin + ray.Direction * t, 1.0f)).xyz;
}

float calcObjectSpaceZ(float3 posV, float3 normal, RayDesc ray, float t)
{
    float3 samplePosV = RayToViewSpace(ray, t);
    float objectSpaceZ = dot(samplePosV - posV, normal);
    return objectSpaceZ;
}

// z: positive linear depth in view space
// r: radius in view/world space
float2 ViewSpaceRadiusToUVRadius(float z, float r)
{
    //const float2 imageScale = 0.5 * float2(gCamera.data.frameWidth / gCamera.data.focalLength, gCamera.data.frameHeight / gCamera.data.focalLength);
    //float2 ndc = float2(r) / (imageScale * z); // radius in normalized device coordinates
    //return ndc * 0.5; // scale to uv radius
    return float2(r * gCamera.data.focalLength) * gData.resolution / (float2(gCamera.data.frameWidth, gCamera.data.frameHeight) * z); // radius in normalized device coordinates
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
    float3 normal;
    float3 tangent; 
    float3 bitangent;
    float3 normalO; 

    float radiusInPixels;
    float rayRasterBlend;

    // returns false if sample needs no shading (background)
    [mutating] bool Init(float2 texC)
    {
        float linearDepth = gDepthTex.SampleLevel(gTextureSampler, texC, 0);
        radiusInPixels = GetAORadiusInPixels(linearDepth);
        rayRasterBlend = saturate((radiusInPixels - gData.ssRadiusFadeEnd) / gData.ssRadiusFadeSize);
        
        if (radiusInPixels < 0.5)
            return false;

        posV = UVToViewSpace(texC, linearDepth);
        posVLength = length(posV);

        // view space normal of current pixel
        float3 normalW = gNormalTex.SampleLevel(gTextureSampler, texC, 0).xyz;
        float3 normalV = mul(float3x3(gCamera.data.viewMat), normalW);
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
    bool requireRay;

    float initialSamplePosLength;
    
    // returns false if the sample is invalid (below hemisphere)
    [mutating] bool Init(float2 texC, BasicAOData data, uint i)
    {
        // random angle on view space disc
        float alpha = (float(i) / NUM_DIRECTIONS) * 2.0 * 3.141;
        float radius = sampleRadius[i] * gData.radius; // radius on sampling unit sphere * world space radius
        float2 dir = radius * float2(sin(alpha), cos(alpha)); // world space direction

        const float sphereHeight = sqrt(gData.radius * gData.radius - radius * radius);
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
        float3 initialSamplePosV = data.posV + data.tangent * dir.x + data.bitangent * dir.y;
        initialSamplePosLength = length(initialSamplePosV);
        samplePosUV = ViewSpaceToUV(initialSamplePosV);
        requireRay = false;
        visibility = 0.0;
        objectSpaceZ = 0.0;
        
        float2 screenUv = getScreenClampedUV(texC, samplePosUV); // clip to screen border
        isInScreen = all(samplePosUV == screenUv);
            
        rasterSamplePosUV = screenUv;
        rasterSamplePosUV = getSnappedUV(rasterSamplePosUV); // snap to pixel center
        
        return true;
    }

    [mutating] void evalPrimaryVisibility(BasicAOData data)
    {
        objectSpaceZ = calcObjectSpaceZ(data.posV, data.normal, rasterSamplePosUV, gDepthTex);
        visibility = calcVisibility(objectSpaceZ, sphereStart, sphereEnd, pdf);
        requireRay = objectSpaceZ > sphereStart + CONST_RADIUS;
    }

    [mutating] void evalDualVisibility(BasicAOData data)
    {
        if (!requireRay)
            return; // not needed
        
        objectSpaceZ = min(objectSpaceZ, calcObjectSpaceZ(data.posV, data.normal, rasterSamplePosUV, gDepthTex2));
        visibility = COMBINE_VIS(visibility, calcVisibility(objectSpaceZ, sphereStart, sphereEnd, pdf));
        requireRay = objectSpaceZ > sphereStart + CONST_RADIUS;
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

float2 calcAO2(uint2 svPos, uint mask)
{
    float2 texC = (float2(svPos) + 0.5) * gData.invResolution;

    BasicAOData data;
    data.Init(texC);

    float2 visibility = 0.0f;

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
            float width, height;
            gDepthTex.GetDimensions(width, height);
            
            int2 pixelCoord = int2(floor(s.rasterSamplePosUV * float2(width, height)));

            const float depthRange = gCamera.data.farZ - gCamera.data.nearZ;
            const float depthOffset = gCamera.data.nearZ;
            bool stochInside = false;
            [unroll]
            for (uint i = 0; i < MSAA_SAMPLES; ++i)
            {
                float linearSampleDepth = LOAD_STOCHASTIC_SAMPLE(gsDepthTex, pixelCoord, i);
                // linearSampleDepth is in [0, 1] => scale accordingly
                linearSampleDepth = linearSampleDepth * depthRange + depthOffset;
                float3 samplePosV = UVToViewSpace(s.rasterSamplePosUV, linearSampleDepth);
                float objectSpaceZ = dot(samplePosV - data.posV, data.normal);
                if (objectSpaceZ <= gData.radius + gData.thickness * gData.radius && objectSpaceZ >= -gData.radius)
                    stochInside = true;
                float newVisibility = calcVisibility(objectSpaceZ, s.sphereStart, s.sphereEnd, s.pdf);
                curVisibility = COMBINE_VIS(curVisibility, newVisibility);
            }
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
            rayData.tLastFrontFaceHalo = (data.posVLength - s.sphereStart - gData.radius - gData.thickness * gData.radius) * s.initialSamplePosLength / data.posVLength; // min (haloStart)
            rayData.tFirstFrontFaceInside = (data.posVLength - s.sphereEnd) * s.initialSamplePosLength / data.posVLength; // max (sphereEnd)
            rayData.tConstRadiusStart = (data.posVLength - gData.radius - gData.thickness * gData.radius) * s.initialSamplePosLength / data.posVLength;
            rayData.tSphereStart = (data.posVLength - s.sphereStart) * s.initialSamplePosLength / data.posVLength;

            ray.TMin = max(rayData.tLastFrontFaceHalo, 0.0);
            ray.TMax = rayData.tFirstFrontFaceInside; // sphereEnd

            const float epsilon = gData.radius * 0.01;
            // include the value of the depth buffer when choosing TMin to save some traversal time
            if (s.isInScreen)
                ray.TMin = max(ray.TMin, (data.posVLength - s.objectSpaceZ) * s.initialSamplePosLength / data.posVLength + epsilon);
            else
                curVisibility = 1.0;
            
            traceAORay(ray, rayData);

            float sphereVisibility = calcVisibility(data.posVLength - rayData.tFirstFrontFaceInside * data.posVLength / s.initialSamplePosLength, s.sphereStart, s.sphereEnd, s.pdf);
            float haloVisibility = calcHaloVisibility(data.posVLength - rayData.tLastFrontFaceHalo * data.posVLength / s.initialSamplePosLength, s.sphereStart, s.sphereEnd, s.pdf);

            float rayVisibility = min(curVisibility, min(sphereVisibility, haloVisibility));
            curVisibility = lerp(s.visibility, rayVisibility, data.rayRasterBlend);
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

// returns any of the above A0_HIT defines
uint aoAnyHit(inout RayData rayData, float t, const TriangleHit hit, bool frontFace)
{
    const uint materialID = gScene.getMaterialID(hit.instanceID);
    const MaterialHeader header = gScene.materials.materialData[materialID].header;

    bool isAlphaTested = header.getAlphaMode() == AlphaMode::Mask;

    // needs alpha testing?
    if (isAlphaTested)
    {
        const VertexData v = gScene.getVertexData(hit);
        if (gScene.materials.alphaTest(v, materialID, 0.0)) // TODO correct lod?   
            return AO_HIT_IGNORE; // alpha test failed => ignore this triangle
    }

    frontFace = frontFace || isAlphaTested || header.isDoubleSided();
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
