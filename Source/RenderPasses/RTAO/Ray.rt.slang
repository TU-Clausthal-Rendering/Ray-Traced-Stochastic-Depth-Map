import Scene.Camera.Camera;
import RTAOData;
import Scene.Intersection;
import Scene.Raytracing;
import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;
import Utils.Math.MathHelpers;


Texture2D<float4> gWPosTex;
Texture2D<float3> gFaceNormalTex;
RWTexture2D<float> ambientOut;
RWTexture2D<float> rayDistanceOut;

Texture1D<float4> gSamples;

cbuffer StaticCB
{
    RTAOData gData;
}

cbuffer PerFrameCB
{
    Camera gCamera;
    uint frameIndex;
}

struct RayData
{
    float t; // ray intersection
};

[shader("miss")]
void miss(inout RayData rayData)
{
    rayData.t = 0;
}

[shader("anyhit")]
void anyHit(inout RayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    // Alpha test for non-opaque geometry.
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    if (gScene.materials.alphaTest(v, materialID, 0.f))
        IgnoreHit();
}

[shader("closesthit")]
void closestHit(inout RayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    rayData.t = RayTCurrent();
}

float3 getTangentVector(float3 n)
{
    return abs(n.x) > abs(n.y) ? float3(-n.z, 0.0, n.x) : float3(0.0, n.z, -n.y); // from pbrt
}

uint jenkins(uint a)
{
    a -= (a << 6);
    a ^= (a >> 17);
    a -= (a << 9);
    a ^= (a << 4);
    a -= (a << 3);
    a ^= (a << 10);
    a ^= (a >> 15);
    return a;
}

uint hash(uint3 c)
{
    // xy = screen position offset, z = frame index
    return jenkins(c.x * 449 + c.y * 2857 + jenkins(c.z));
}

float CalculateAO(in float tHit)
{
    float ambientCoef = 1.f;
    if (tHit > 0)
    {
        float occlusionCoef = 1;
        if (gData.applyExponentialFalloff)
        {
            float t = tHit / gData.maxTheoreticalTHit;
            float lamda = gData.exponentialFalloffDecayConstant;
            occlusionCoef = exp(-lamda * t * t);
        }

        ambientCoef = 1 - (1 - gData.minimumAmbientIllumination) * occlusionCoef;
    }

    return ambientCoef;
}

float AOOnce(out float tHit, in float3 wPos, in float3 faceNormal)
{
    const uint2 svPos = DispatchRaysIndex().xy;

    // determine tangent space
    float3 bitangent = getTangentVector(faceNormal);
    float3 tangent = cross(bitangent, faceNormal);
 
    // onbtain a single ray sample that is cosine distributed
    uint numSamples;
    gSamples.GetDimensions(numSamples);
    float3 randDir = gSamples[hash(uint3(svPos, frameIndex)) % numSamples].xyz;

    RayDesc ray;
    ray.Origin = wPos.xyz;
    ray.Direction = normalize(tangent * randDir.x + bitangent * randDir.y + faceNormal * randDir.z);
    ray.Origin += faceNormal * gData.normalScale; // push origin in the direction of the face normal to avoid self intersection
    ray.TMin = 0.001;
    ray.TMax = gData.maxAORayTHit;

    RayData rayData;
    rayData.t = 0.0; //set to miss
    TraceRay(
        gScene.rtAccel,
        RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES, // | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
        0xff /* instanceInclusionMask */, 0 /* hitIdx */, 1 /*ray type count*/, 0 /* missIdx */,
        ray, rayData
    );

    tHit = rayData.t > 0 ? rayData.t : gData.maxAORayTHit;
    return CalculateAO(rayData.t);
}



//Has its own function as a different sample generator is used
float AOMoreThanOnce(out float tHit, in float3 wPos, in float3 faceNormal)
{
    const uint2 svPos = DispatchRaysIndex().xy;

    // determine tangent space
    float3 bitangent = getTangentVector(faceNormal);
    float3 tangent = cross(bitangent, faceNormal);

    //Init sample generator
    SampleGenerator sg = SampleGenerator(svPos, frameIndex);

    RayDesc ray;
    ray.Origin = wPos.xyz;
    ray.Origin += faceNormal * gData.normalScale; // push origin in the direction of the face normal to avoid self intersection
    ray.TMin = 0.001;
    ray.TMax = gData.maxAORayTHit;

    RayData rayData;

    float ambientCoef = 0;
    tHit = gData.maxAORayTHit;

    for (uint i = 0; i < gData.spp; i++)
    {
        float pdf;  //needed by the sample function
        float3 randDir = sample_cosine_hemisphere_polar(sampleNext2D(sg), pdf);
        ray.Direction = normalize(tangent * randDir.x + bitangent * randDir.y + faceNormal * randDir.z);

        rayData.t = 0.0; //set to miss

        TraceRay(
        gScene.rtAccel,
        RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES, // | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
        0xff /* instanceInclusionMask */, 0 /* hitIdx */, 1 /*ray type count*/, 0 /* missIdx */,
        ray, rayData
        );

        ambientCoef += CalculateAO(rayData.t);
        tHit += rayData.t > 0 ? rayData.t : gData.maxAORayTHit;
    }

    ambientCoef /= gData.spp;
    tHit /= gData.spp;
    
    return ambientCoef;
}

[shader("raygeneration")]
void rayGen()
{
    const uint2 svPos = DispatchRaysIndex().xy;
    
    // non-linear depth
    float4 wPos = gWPosTex[svPos];
    if (wPos.w == 0.0) //0 in w is miss
    {
        rayDistanceOut[svPos] = gData.maxAORayTHit;
        ambientOut[svPos] = 1;
        return;
    }

    // view space normal of current pixel
    float3 faceNormal = normalize(gFaceNormalTex[svPos]);

    float tHit = 0.0;
    float ambientCoef = 1;

    if (gData.spp > 1)
    {
        ambientCoef = AOMoreThanOnce(tHit, wPos.xyz, faceNormal);
    }
    else
    {
        ambientCoef = AOOnce(tHit, wPos.xyz, faceNormal);
    }

    rayDistanceOut[svPos] = tHit;
    ambientOut[svPos] = ambientCoef;
}
