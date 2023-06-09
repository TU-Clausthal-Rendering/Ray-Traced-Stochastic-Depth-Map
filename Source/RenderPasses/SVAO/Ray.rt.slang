import Scene.Raytracing;
#include "Common.slang"

// inputs from previous stage
Texture2D<uint> aoMask;
RWTexture2D<float> output;

[shader("anyhit")]
void anyHit(inout RayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    // extract hit properties
    float t = RayTCurrent();
    if (t < rayData.tLastFrontFaceHalo)
        IgnoreHit(); // we can skip this since it would not contribute anyways (unfortunately there is no option to set the ray.min afterwards)

    bool frontFace = HitKind() == HIT_KIND_TRIANGLE_FRONT_FACE;
    TriangleHit hit;
    hit.instanceID = getGeometryInstanceID();
    hit.primitiveIndex = PrimitiveIndex();
    hit.barycentrics = attribs.barycentrics;
    const uint materialID = gScene.getMaterialID(hit.instanceID);
    const MaterialHeader header = gScene.materials.materialData[materialID].header;

    bool isAlphaTested = header.getAlphaMode() == AlphaMode::Mask;

    // needs alpha testing?
    if (isAlphaTested)
    {
        const VertexData v = gScene.getVertexData(hit);
        if (gScene.materials.alphaTest(v, materialID, 0.0)) // TODO correct lod?   
            IgnoreHit(); // alpha test failed => ignore this triangle
    }

    frontFace = frontFace || isAlphaTested || header.isDoubleSided();
    if (!frontFace)
        IgnoreHit(); // this is just for rasterizer compability

    if (t <= rayData.tSphereStart)
    {
        rayData.tLastFrontFaceHalo = max(rayData.tLastFrontFaceHalo, t);
        if (t >= rayData.tConstRadiusStart)
            AcceptHitAndEndSearch(); // we can stop the query, because this will set the visibility to zero
    }
    else // inside sphere
    {
        rayData.tFirstFrontFaceInside = min(rayData.tFirstFrontFaceInside, t);
        return; // since we save the min, we can commit TMax here
    }
    IgnoreHit(); // continue traversal
}

[shader("miss")]
void miss(inout RayData rayData)
{
}

[shader("closesthit")]
void closestHit(inout RayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    // is skipped
}

// implementation of the traceAORay used inside calcAO2
void traceAORay(RayDesc ray, inout RayData rayData)
{
    TraceRay(
        gScene.rtAccel,
        RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_FORCE_NON_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
        0xff /* instanceInclusionMask */, 0 /* hitIdx */, 1 /*ray type count*/, 0 /* missIdx */,
        ray, rayData
    );
}

[shader("raygeneration")]
void rayGen()
{
    uint2 svPos = DispatchRaysIndex().xy + uint2(guardBand);
    float2 texC = (float2(svPos) + 0.5) * gData.invResolution;

    uint mask = aoMask[svPos] & 0xFF;
    if (mask == 0u)
        return;

    float visibility = calcAO2(svPos, mask);
    visibility += output[svPos];
    visibility = pow(visibility, gData.exponent);
    output[svPos] = visibility;
}
