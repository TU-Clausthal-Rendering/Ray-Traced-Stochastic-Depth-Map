import Scene.Raytracing;

Texture2D<float> depthInTex;
RWTexture2DArray<float> depthOutTex;
Texture2D<uint> maskTex;

StructuredBuffer<uint> stratifiedIndices;
StructuredBuffer<uint> stratifiedLookUpTable;

// Hash function from "Improved Alpha Testing Using Hashed Sampling"
float hash(float2 v)
{
    return frac(1.0e4 * sin(17.0 * v.x + 0.1 * v.y) * (0.1 + abs(sin(13.0 * v.y + v.x))));
}

// 3D hash function from "Improved Alpha Testing Using Hashed Sampling"
float hash3D(float3 v)
{
    return hash(float2(hash(v.xy), v.z));
}

// 4D hash function based on the 3D hash function
float hash4D(float4 v)
{
    return hash(float2(hash3D(v.xyz), v.w));
}

struct RayData // cannot be compressed to half floats => no diff in rendering time + insufficient visual quality
{
    float depths[NUM_SAMPLES];
};

float RayToViewDepth(float3 rayDir, float rayT)
{
    float cosTheta = dot(normalize(gScene.camera.data.cameraW), rayDir);
    return rayT * cosTheta;
}

float ViewDepthToRay(float3 rayDir, float viewDepth)
{
    float cosTheta = dot(normalize(gScene.camera.data.cameraW), rayDir);
    return viewDepth / cosTheta;    
}

[shader("anyhit")]
void anyHit(inout RayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    TriangleHit hit;
    hit.instanceID = getGeometryInstanceID();
    hit.primitiveIndex = PrimitiveIndex();
    hit.barycentrics = attribs.barycentrics;
    const uint materialID = gScene.getMaterialID(hit.instanceID);
    const MaterialHeader header = gScene.materials.materialData[materialID].header;

    // needs alpha testing?
    if (header.getAlphaMode() == AlphaMode::Mask)
    {
        const VertexData v = gScene.getVertexData(hit);
        if (gScene.materials.alphaTest(v, materialID, 0.0)) // TODO correct lod?   
            IgnoreHit(); // alpha test failed => ignore this triangle
    }

    float t = RayTCurrent();
    t = RayToViewDepth(WorldRayDirection(), t); // convert to view depth
    
    //float rng = hash4D(float4(DispatchRaysIndex().x, DispatchRaysIndex().y, t, 1.438943289));
    float rng = hash4D(float4(attribs.barycentrics, t, 1.438943289));

    int R = int(floor(ALPHA * NUM_SAMPLES + rng));
    uint coverageMask = 0;
    
    if (R >= NUM_SAMPLES)
    {
        coverageMask = 0xffff; // maximum 32 samples
    }
    else if (R != 0)
    {
        float rng2 = hash3D(float3(attribs.barycentrics, t));
        int index = int(lerp(stratifiedIndices[R], stratifiedIndices[R + 1], rng2));
        coverageMask = stratifiedLookUpTable[index];
    }

    if(NORMALIZE)
    {
        t = saturate((t - gScene.camera.data.nearZ) / (gScene.camera.data.farZ - gScene.camera.data.nearZ));
    }

    float maxT = 0.0;
    [unroll]
    for (uint i = 0; i < NUM_SAMPLES; ++i)
    {
        if (coverageMask & (1 << i))
            rayData.depths[i] = min(rayData.depths[i], t); // z-test enabled
        maxT = max(maxT, rayData.depths[i]);
    }

    if (t < maxT)
        IgnoreHit(); // keep looking for hits, otherwise start to terminate
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

[shader("raygeneration")]
void rayGen()
{
    uint2 svPos = DispatchRaysIndex().xy;

    uint mask = maskTex[svPos] & 0xFF;
    if (mask == 0u)
        return;

    RayDesc ray = gScene.camera.computeRayPinhole(DispatchRaysIndex().xy, DispatchRaysDimensions().xy).toRayDesc();
    float epsilon = 0.01 * gScene.camera.data.nearZ; // small ray offset
    ray.TMin = ViewDepthToRay(ray.Direction, depthInTex[svPos]) + epsilon; // start after first known hit
    
    // TODO offset ray start to depth in
    RayData rayData;
    for (uint i = 0; i < NUM_SAMPLES; ++i)
        rayData.depths[i] = NORMALIZE ? 1.0 : 3.40282347E+37F;
    
    // TODO custom culling?
    TraceRay(
        gScene.rtAccel,
        RAY_FLAG_CULL_BACK_FACING_TRIANGLES | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_FORCE_NON_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
        0xff /* instanceInclusionMask */, 0 /* hitIdx */, 1 /*ray type count*/, 0 /* missIdx */,
        ray, rayData
    );

    for (uint i = 0; i < NUM_SAMPLES; ++i)
        depthOutTex[uint3(svPos, i)] = rayData.depths[i];
}
