import Scene.RaytracingInline;
#include "Common.slang"
#include "NeuralNetDefines.slangh"
#include "NeuralNetDefines2.slangh"

struct PSOut
{
    float ao1 : SV_TARGET0;
    uint stencil : SV_TARGET1;

};

RWTexture2D<uint> gDepthAccess; // indicates depth accesses for the secondary pass


PSOut main(float2 texC : TEXCOORD, float4 svPos : SV_POSITION)
{
    PSOut output;
    output.ao1 = 0.0;
    output.stencil = 0;

    float linearDepth = gDepthTex.SampleLevel(gTextureSampler, texC, 0);
    float radiusInPixels = GetAORadiusInPixels(linearDepth);

    //if (linearDepth >= gCamera.data.farZ)
    if(radiusInPixels < 0.5)
    {
        output.ao1 = 1.0;
        return output;
    }

    //float rayRasterBlend = saturate((radiusInPixels - gData.ssRadiusFadeEnd) / max(gData.ssRadiusFaceStart - gData.ssRadiusFaceEnd, 0.01));
    bool denyRays = radiusInPixels <= gData.ssRadiusFadeEnd;
    //bool denyRays = false;
    
    // view space position of current pixel
    const float3 posV = UVToViewSpace(texC, linearDepth);
    const float posVLength = length(posV);

    // view space normal of current pixel
    float3 normalW = gNormalTex.SampleLevel(gTextureSampler, texC, 0).xyz;
    float3 normalV = mul(normalW, float3x3(gCamera.data.viewMat));
    if (dot(posV, normalV) > 0.0)
        normalV = -normalV;

    // Calculate tangent space (use random direction for tangent orientation)
    float randRotation = gNoiseTex.SampleLevel(gNoiseSampler, texC * gData.noiseScale, 0) * 2.0 * 3.141;
    float2 randDir = float2(sin(randRotation), cos(randRotation));
    //randDir = float2(1.0f, 0.0f);
    
    // determine tangent space
    float3 normal = -posV / posVLength;
    float3 bitangent = normalize(cross(normal, float3(randDir, 0.0f)));
    float3 tangent = cross(bitangent, normal);

    // transfer view space normal to normal in object coordinates of the sampling sphere
    float3 normalO = float3(dot(normalV, tangent), dot(normalV, bitangent), dot(normalV, normal));
    
    float visibility = 0.0f;

    #if PRIMARY_DEPTH_MODE >= DEPTH_MODE_MACHINE_CLASSIFY
    float rasterHeights[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    #endif

    uint forceRayMask = 0; // forces rays due to double sided materials
    bool isDoubleSided = (gMatDoubleSided[UVToPixel(texC)] & MTL_DOUBLE_SIDED) != 0;
    
    [unroll] for (uint i = 0; i < NUM_DIRECTIONS; i++)
    {
        // random angle on view space disc
        float alpha = (float(i) / NUM_DIRECTIONS) * 2.0 * 3.141;
        float radius = sampleRadius[i] * gData.radius; // radius on sampling unit sphere * world space radius
        float2 dir = radius * float2(sin(alpha), cos(alpha));  // world space direction

        const float sphereHeight = sqrt(gData.radius * gData.radius - radius * radius);
        const float pdf = 2.0 * sphereHeight;
        
        // determine distance within [-sphereHeight, +sphereHeight]
        float sphereStart = sphereHeight; // in object coordinates (bigger is closer to the camera)
        float sphereEnd = -sphereHeight; // in object coordinates (smaller is futher from the camera)

        { // HEMISPHERE SAMPLING
            //float zIntersect = -dot(rand.xy, normalO.xy) / normalO.z;
            float zIntersect = -dot(dir.xy, normalO.xy) / makeNonZero(normalO.z, 0.0001);
            float zIntersectClamped = clamp(zIntersect, -sphereHeight, sphereHeight);
            sphereEnd = zIntersectClamped;
        }
        
        // sample position calculate uv position of sample
        float3 initialSamplePosV = posV + tangent * dir.x + bitangent * dir.y;
        float2 samplePosUV = ViewSpaceToUV(initialSamplePosV);

        float curVisibility = 0.0f;
        bool requireRay = false;
        bool forceRay = false;

        float2 screenUv = getScreenClampedUV(texC, samplePosUV); // clip to screen border
        if (any(samplePosUV != screenUv) && SECONDARY_DEPTH_MODE == DEPTH_MODE_RAYTRACING)
        {
            if (TRACE_OUT_OF_SCREEN)
                forceRay = true; // always shoot rays for screen border   
        }
            
        float2 rasterSamplePosUV = screenUv;
        rasterSamplePosUV = getSnappedUV(rasterSamplePosUV); // snap to pixel center

        if (isSamePixel(texC, rasterSamplePosUV))
        {
            output.ao1 += (sphereStart - sphereEnd) / pdf;
            continue;
        }
        
        float objectSpaceZ = calcObjectSpaceZ(posV, normal, rasterSamplePosUV, gDepthTex);
        curVisibility = calcVisibility(objectSpaceZ, sphereStart, sphereEnd, pdf);

        #if PRIMARY_DEPTH_MODE >= DEPTH_MODE_MACHINE_CLASSIFY
        rasterHeights[i] = clamp(objectSpaceZ / gData.radius, -16.0, 16.0);
        #endif



        if ((sphereStart - sphereEnd) / pdf <= 0.1)
        {
            continue; // skip sample (no visibility)
        }
        
        // require hybrid ray if intersection is outside of sphere radius
        if (!denyRays && PRIMARY_DEPTH_MODE != DEPTH_MODE_DUAL && objectSpaceZ > sphereStart + CONST_RADIUS)
        {
            bool hitDoubleSided = (gMatDoubleSided[UVToPixel(rasterSamplePosUV)] & MTL_DOUBLE_SIDED) != 0;
        // require ray unless we dont want double sided hit and both materials are double sided
            if (TRACE_DOUBLE_ON_DOUBLE || !isDoubleSided || !hitDoubleSided)
            {
                requireRay = true;
#if PRIMARY_DEPTH_MODE >= DEPTH_MODE_MACHINE_CLASSIFY
        // fetch material data to check for double sided flag
        if (hitDoubleSided)
            forceRay = true;
#endif    
            }
            
        }

        if (PRIMARY_DEPTH_MODE == DEPTH_MODE_DUAL && objectSpaceZ > sphereStart + CONST_RADIUS)
        {
            float objectSpaceZ = calcObjectSpaceZ(posV, normal, rasterSamplePosUV, gDepthTex2);
            curVisibility = COMBINE_VIS(curVisibility, calcVisibility(objectSpaceZ, sphereStart, sphereEnd, pdf));
            if (objectSpaceZ > sphereStart + CONST_RADIUS) // both samples were before the sphere starts
                requireRay = true;
        }

        // always add computed visibility from raster
        output.ao1 += curVisibility;
        if (denyRays)
            continue;
        
        if (forceRay && (PRIMARY_DEPTH_MODE >= DEPTH_MODE_MACHINE_CLASSIFY))
        {
            forceRayMask |= (1u << i);
        }
        else if (requireRay || forceRay)
        {
        // remember that this was a hybrid ray
            if (PRIMARY_DEPTH_MODE != DEPTH_MODE_PERFECT_CLASSIFY)
                output.stencil |= 1u << i;
            
#if SECONDARY_DEPTH_MODE == DEPTH_MODE_STOCHASTIC
            // remember that an access was requested
            gDepthAccess[uint2(rasterSamplePosUV * gData.resolution)] = 1u;
#endif
#if PRIMARY_DEPTH_MODE == DEPTH_MODE_PERFECT_CLASSIFY
            // determine ray t to decide if a ray is necessary
            samplePosUV = getSnappedUV(samplePosUV); // snap to pixel center
            
            float3 sampleDirV = normalize(UVToViewSpace(samplePosUV, 1.0)); // get sample direction in view space
            float initialSamplePosLength = length(initialSamplePosV);

            RayDesc ray;
            ray.Origin = gCamera.data.posW;
            ray.Direction = mul(sampleDirV, float3x3(invViewMat));


            if (PREVENT_DARK_HALOS)
            {
                // only trace the ray inside the halo area
                float tSphereStart = (posVLength - sphereStart) * initialSamplePosLength / posVLength;
                float tSphereEnd = (posVLength - sphereEnd) * initialSamplePosLength / posVLength;
                float tHaloStart = (posVLength - sphereHeight - CONST_RADIUS - HALO_RADIUS) * initialSamplePosLength / posVLength;
                float tConstRadiusStart = (posVLength - sphereHeight - CONST_RADIUS) * initialSamplePosLength / posVLength;
                float tExtendedRadiusStart = (posVLength - 2 * gData.radius) * initialSamplePosLength / posVLength;
            
                ray.TMin = 0.0; // include whole range for ML learning
                ray.TMax = gCamera.data.farZ;

                // skip procedural and force all triangle to be handled by any-hit traversal
                RayQuery < RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_FORCE_NON_OPAQUE > rayQuery;
                rayQuery.TraceRayInline(gScene.rtAccel, RAY_FLAG_NONE, 0xff, ray);

                float tLastFrontFaceHalo = ray.TMin;
                float tFirstFrontFace = ray.TMax;
                float tFirstBackface = ray.TMax;
                float tFirstFrontFaceInside = ray.TMax;
                bool firstFrontFaceDoubleSided = false;

                while (rayQuery.Proceed())
                {
                    if (rayQuery.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
                    {
                    // extract hit properties
                        float t = rayQuery.CandidateTriangleRayT();

                        bool frontFace = rayQuery.CandidateTriangleFrontFace();
                        const TriangleHit hit = getCandidateTriangleHit(rayQuery);
                        const uint materialID = gScene.getMaterialID(hit.instanceID);
                        const MaterialHeader header = gScene.materials.materialData[materialID].header;
                        bool isAlphaTested = header.getAlphaMode() == AlphaMode::Mask;

                    // needs alpha testing?
                        if (isAlphaTested)
                        {
                            const VertexData v = gScene.getVertexData(hit);
                            if (gScene.materials.alphaTest(v, materialID, 0.0)) // TODO correct lod?   
                                continue; // alpha test failed => ignore this triangle
                        }

                        frontFace = frontFace || isAlphaTested || header.isDoubleSided();
                        if (!frontFace)
                        {
                            tFirstBackface = min(tFirstBackface, t);
                            if (tFirstFrontFaceInside < t)
                                rayQuery.CommitNonOpaqueTriangleHit(); // since we save the min, we can commit here
                            continue;
                        }
                        
                        tFirstFrontFace = min(tFirstFrontFace, t);
                        if (tFirstFrontFace == t)
                            firstFrontFaceDoubleSided = header.isDoubleSided();

                        if (t <= tSphereStart)
                        {
                            tLastFrontFaceHalo = max(tLastFrontFaceHalo, t);
                            //if (t >= tConstRadiusStart)
                            //    break; // we can stop the query, because this will set the visibility to zero
                        }
                        else // inside sphere
                        {
                            tFirstFrontFaceInside = min(tFirstFrontFaceInside, t);
                            if (tFirstBackface < t)
                                rayQuery.CommitNonOpaqueTriangleHit(); // since we save the min, we can commit here
                        }
                        
                    }
                }

                //objectSpaceZ = (posVLength - tFirstFrontFace * posVLength / initialSamplePosLength);
                float rayDepth = objectSpaceZ;

                bool requireRay = true;
                if(!firstFrontFaceDoubleSided)
                {
                    if (tFirstFrontFace >= tConstRadiusStart) requireRay = false;
                    if (tFirstBackface >= tConstRadiusStart) requireRay = false;
                    float tGrazingAngle = 1.0 - (sphereStart - sphereEnd)/pdf;
                    float tExtended = lerp(tConstRadiusStart, tExtendedRadiusStart, saturate(saturate(tGrazingAngle - 0.5) / 0.2));
                    if (tFirstBackface >= tExtended)  requireRay = false;
                }

                // determine if ray is required
                if(requireRay)
                    output.stencil |= 1u << i; // still needs a ray
            }
#endif
        }
    }

    #if PRIMARY_DEPTH_MODE == DEPTH_MODE_MACHINE_CLASSIFY
    if (output.stencil != 0)
    {
        uint machineMask = evalClassifier(rasterHeights, gData.classifyThreshold);
        output.stencil = output.stencil & machineMask;
    }
    output.stencil |= forceRayMask; // always trace the forced rays
    #endif
    
    #if PRIMARY_DEPTH_MODE == DEPTH_MODE_PERFECT_CLASSIFY
    output.stencil |= forceRayMask; // always trace the forced rays
    #endif
    
    #if PRIMARY_DEPTH_MODE == DEPTH_MODE_MACHINE_PREDICT
    if(output.stencil != 0)
    {
        //output.stencil |= forceRayMask;
        evalRegressor(rasterHeights);

        [unroll]
        for (uint i = 0; i < NUM_DIRECTIONS; ++i)
        {
            if ((output.stencil & (1u << i)))
            {
                // calc new ao value from new height
                float alpha = (float(i) / NUM_DIRECTIONS) * 2.0 * 3.141;
                float radius = sampleRadius[i] * gData.radius; // radius on sampling unit sphere * world space radius
                float2 dir = radius * float2(sin(alpha), cos(alpha)); // world space direction

                const float sphereHeight = sqrt(gData.radius * gData.radius - radius * radius);
                const float pdf = 2.0 * sphereHeight;
        
                // determine distance within [-sphereHeight, +sphereHeight]
                float sphereStart = sphereHeight; // in object coordinates (bigger is closer to the camera)
                float sphereEnd = -sphereHeight; // in object coordinates (smaller is futher from the camera)

                float zIntersect = -dot(dir.xy, normalO.xy) / makeNonZero(normalO.z, 0.0001);
                float zIntersectClamped = clamp(zIntersect, -sphereHeight, sphereHeight);
                sphereEnd = zIntersectClamped;

                float objectSpaceZ = rasterHeights[i];
                float curVisibility = calcVisibility(objectSpaceZ, sphereStart, sphereEnd, pdf);
                output.ao1 += curVisibility;
            }
        }
    }
    output.stencil = forceRayMask; // replace with forceRayMask
    #endif

    output.ao1 *= 2.0 / float(NUM_DIRECTIONS);

    //if (countbits(output.stencil) == 1)
    //    output.stencil = 0;
   

    if (RAY_FILTER)
    {
        uint numWaves = WaveActiveCountBits(true);

        // count of all other threads (not including self)
        //uint otherRayCount = WaveActiveSum(countbits(output.stencil)) - countbits(output.stencil);
        uint otherRayCount = WaveActiveSum(output.stencil != 0 ? 1 : 0);

        float threshold = 0.2f;
        //if (otherRayCount < uint(threshold * numWaves))
        //    output.stencil = 0;

        //if (WaveActiveMax(countbits(output.stencil)) <= 1)
        //    output.stencil = 0;

        if (countbits(output.stencil) == 1
            && QuadReadAcrossX(output.stencil) == 0
            && QuadReadAcrossY(output.stencil) == 0
            && QuadReadAcrossDiagonal(output.stencil) == 0)
            output.stencil = 0;

    }

    // apply exponent
    if(output.stencil == 0)
        output.ao1 = pow(output.ao1, gData.exponent);
    
    return output;
}
