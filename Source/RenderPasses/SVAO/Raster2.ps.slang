import Scene.RaytracingInline;
#include "Common.slang"

// additional depth textures
Texture2DMS<float> gsDepthTex;

// inputs from previous stage
Texture2D<uint> aoMask;
Texture2D<float> aoPrev;

[earlydepthstencil]
float main(float2 texC : TEXCOORD, float4 svPos : SV_POSITION) : SV_TARGET
{
    uint mask = aoMask[uint2(svPos.xy)] & 0xFF;

    // fetch linear depth a second time
    float linearDepth = gDepthTex.SampleLevel(gTextureSampler, texC, 0);
    float radiusInPixels = GetAORadiusInPixels(linearDepth);
    float rayRasterBlend = saturate((radiusInPixels - gData.ssRadiusFadeEnd) / gData.ssRadiusFadeSize);
    
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

    bool isDoubleSided = (gMatDoubleSided[UVToPixel(texC)] & MTL_DOUBLE_SIDED) != 0;
    
    uint i = 0;
    //[loop] while(mask != 0u)
    [unroll] for (uint j = 0; j < NUM_DIRECTIONS; j++)
    {
        if (mask == 0u) break; // no bits set anymore

        // modify loop to only go through the set bits in mask
        //[loop] while ((mask & 1u) == 0u)
        //for (uint k = 0; k < (NUM_DIRECTIONS - j) && (mask & 1u) == 0u; k++) // this is too complicated for current compiler..  
        [unroll] for (uint k = 0; k < NUM_DIRECTIONS && k < NUM_DIRECTIONS - j && (mask & 1u) == 0u; k++) // first condition is for unrolling, second is for better unrolling
        {
            // shift mask an increase i
            mask = mask >> 1;
            ++i;
        }


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

        // if the sample range is too small, skip calculation (sample could be entirely below the surface hemisphere when looking from grazing angles)
        if (sphereStart - sphereEnd < 0.01)
        {
            continue; // skip sample (no visibility)
        }

        // sample position calculate uv position of sample
        float3 initialSamplePosV = posV + tangent * dir.x + bitangent * dir.y;
        float2 samplePosUV = ViewSpaceToUV(initialSamplePosV);
        float curVisibility = 1.0f;

        float2 screenUv = getScreenClampedUV(texC, samplePosUV); // clip to screen border
        const bool isInScreen = all(samplePosUV == screenUv);

        float2 rasterSamplePosUV = screenUv;
        rasterSamplePosUV = getSnappedUV(rasterSamplePosUV); // snap to pixel center

        if (SECONDARY_DEPTH_MODE == DEPTH_MODE_STOCHASTIC) 
        {
            // reuse old depth (this does not really cost much)
            if (PRIMARY_DEPTH_MODE != DEPTH_MODE_DUAL) // DEPTH_MODE = SINGLE (or classify)
            {
                float objectSpaceZ = calcObjectSpaceZ(posV, normal, rasterSamplePosUV, gDepthTex);
                curVisibility = calcVisibility(objectSpaceZ, sphereStart, sphereEnd, pdf);
            }
            else //if (PRIMARY_DEPTH_MODE == DEPTH_MODE_DUAL)
            {
                float objectSpaceZ = calcObjectSpaceZ(posV, normal, rasterSamplePosUV, gDepthTex2);
                curVisibility = calcVisibility(objectSpaceZ, sphereStart, sphereEnd, pdf);
            }

            // subtract old visibility from raster (will be replaced with new visibility)
            visibility -= curVisibility;
            
            int2 pixelCoord = UVToPixel(rasterSamplePosUV);

            const float depthRange = gCamera.data.farZ - gCamera.data.nearZ;
            const float depthOffset = gCamera.data.nearZ;
            [unroll]
            for (uint i = 0; i < MSAA_SAMPLES; ++i)
            {
                float linearSampleDepth = gsDepthTex.Load(pixelCoord, i);
                // linearSampleDepth is in [0, 1] => scale accordingly
                linearSampleDepth = linearSampleDepth * depthRange + depthOffset;
                float3 samplePosV = UVToViewSpace(rasterSamplePosUV, linearSampleDepth);
                float objectSpaceZ = dot(samplePosV - posV, normal);
                float newVisibility = calcVisibility(objectSpaceZ, sphereStart, sphereEnd, pdf);
                curVisibility = COMBINE_VIS(curVisibility, newVisibility);
            }       
        }
        else if (SECONDARY_DEPTH_MODE == DEPTH_MODE_RAYTRACING)
        {
            // to be consistent with the rasterizer, we snap the uv coordinate as well to the pixel center,
            // but we do not clip it since we can shoot outside of the screen space
            //samplePosUV = getScreenClampedUV(texC, samplePosUV);
            samplePosUV = getSnappedUV(samplePosUV); // snap to pixel center
            
            float3 sampleDirV = normalize(UVToViewSpace(samplePosUV, 1.0)); // get sample direction in view space
            float initialSamplePosLength = length(initialSamplePosV);
            
            RayDesc ray;
            ray.Origin = gCamera.data.posW; // offset a little bit in normal direction
            ray.Direction = mul(sampleDirV, float3x3(invViewMat));

            if (PREVENT_DARK_HALOS)
            {
                float tSphereStart = (posVLength - sphereStart) * initialSamplePosLength / posVLength;
                float tSphereEnd = (posVLength - sphereEnd) * initialSamplePosLength / posVLength;
                float tHaloStart = (posVLength - sphereHeight - CONST_RADIUS - HALO_RADIUS) * initialSamplePosLength / posVLength;
                float tConstRadiusStart = (posVLength - sphereHeight - CONST_RADIUS) * initialSamplePosLength / posVLength;
                ray.TMin = max(tHaloStart, 0.0); // TODO include buffer value?
                ray.TMax = tSphereEnd;

                const float epsilon = gData.radius * 0.01;
                float objectSpaceZ;
                if (PRIMARY_DEPTH_MODE != DEPTH_MODE_DUAL) // DEPTH_MODE = SINGLE (or classify)
                    objectSpaceZ = calcObjectSpaceZ(posV, normal, samplePosUV, gDepthTex);
                else // if(PRIMARY_DEPTH_MODE == DEPTH_MODE_DUAL)
                    objectSpaceZ = calcObjectSpaceZ(posV, normal, samplePosUV, gDepthTex2);
                curVisibility = calcVisibility(objectSpaceZ, sphereStart, sphereEnd, pdf);

                // subtract old visibility from raster (will be replaced with new visibility)
                visibility -= curVisibility;

                // include the value of the depth buffer when choosing TMin to save some traversal time
                if (isInScreen) ray.TMin = max(ray.TMin, (posVLength - objectSpaceZ) * initialSamplePosLength / posVLength + epsilon);
                else curVisibility = 1.0;

                // skip procedural and force all triangle to be handled by any-hit traversal
                RayQuery < RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_FORCE_NON_OPAQUE > rayQuery;
                rayQuery.TraceRayInline(gScene.rtAccel, RAY_FLAG_NONE, 0xff, ray);

                float tFirstFrontFaceInside = tSphereEnd;
                float tLastFrontFaceHalo = tHaloStart;

                while (rayQuery.Proceed())
                {
                    if (rayQuery.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
                    {
                    // extract hit properties
                        float t = rayQuery.CandidateTriangleRayT();
                        if (t < tLastFrontFaceHalo)
                            continue; // we can skip this since it would not contribute anyways (unfortunately there is no option to set the ray.min afterwards)

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
                            continue; // this is just for rasterizer compability

                        if (isDoubleSided && !TRACE_DOUBLE_ON_DOUBLE)
                            rayQuery.CommitNonOpaqueTriangleHit(); // always commit in this case
                        
                        if (t <= tSphereStart)
                        {
                            tLastFrontFaceHalo = max(tLastFrontFaceHalo, t);
                            if (t >= tConstRadiusStart)
                                break; // we can stop the query, because this will set the visibility to zero
                        }
                        else // inside sphere
                        {
                            tFirstFrontFaceInside = min(tFirstFrontFaceInside, t);
                            rayQuery.CommitNonOpaqueTriangleHit(); // since we save the min, we can commit here
                        }
                        
                    }
                }

                // calculate visibility inside and outside of sphere
                float sphereVisibility = calcSphereVisibility(posVLength - tFirstFrontFaceInside * posVLength / initialSamplePosLength, sphereStart, sphereEnd, pdf);
                float haloVisibility = calcHaloVisibility(posVLength - tLastFrontFaceHalo * posVLength / initialSamplePosLength, sphereStart, sphereEnd, pdf);

                float rayVisibility = min(curVisibility, min(sphereVisibility, haloVisibility));
                if (isDoubleSided && !TRACE_DOUBLE_ON_DOUBLE)
                    rayVisibility = min(curVisibility, haloVisibility < 1.0 ? haloVisibility : rayVisibility);
                curVisibility = lerp(curVisibility, rayVisibility, rayRasterBlend);
            }
            else // !PREVENT_DARK_HALOS
            {
                float tSphereStart = (posVLength - sphereStart) * initialSamplePosLength / posVLength;
                float tSphereEnd = (posVLength - sphereEnd) * initialSamplePosLength / posVLength;
                ray.TMin = 0.0f;
                ray.TMax = tSphereEnd;
                int occlusionStack = 0; // > 1 means occluded (counted for outside front faces)
                
                if (isInScreen) // start slighlty before the first detected object from the raster pass (start before the object to include it in occlusion mask)
                {
                    const float epsilon = gData.radius * 0.01;
                    //const float epsilon = 0.0;
                    float objectSpaceZ;
                    if (PRIMARY_DEPTH_MODE != DEPTH_MODE_DUAL) // single or classify
                        objectSpaceZ = calcObjectSpaceZ(posV, normal, samplePosUV, gDepthTex);
                    else if(PRIMARY_DEPTH_MODE == DEPTH_MODE_DUAL)
                        objectSpaceZ = calcObjectSpaceZ(posV, normal, samplePosUV, gDepthTex2);
                    ray.TMin = max(ray.TMin, (posVLength - objectSpaceZ) * initialSamplePosLength / posVLength - epsilon);
                }

                // skip procedural and force all triangle to be handled by any-hit traversal
                RayQuery < RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_FORCE_NON_OPAQUE > rayQuery;
                rayQuery.TraceRayInline(gScene.rtAccel, RAY_FLAG_NONE, 0xff, ray);

                float tFirstFrontFaceInside = ray.TMax;


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

                        if (t < tSphereStart) // in front of sphere
                        {
                            if (isAlphaTested || header.isDoubleSided()) // ignore alpha tested materials (too thin for occlusion at this distance)
                                continue;
                            occlusionStack += frontFace ? 1 : -1;
                        }
                        else // inside the sphere
                        {
                            // needs alpha testing?
                            if (isAlphaTested)
                            {
                                const VertexData v = gScene.getVertexData(hit);
                                if (gScene.materials.alphaTest(v, materialID, 0.0)) // TODO correct lod?   
                                    continue; // alpha test failed => ignore this triangle
                            }

                            tFirstFrontFaceInside = min(tFirstFrontFaceInside, t);
                            rayQuery.CommitNonOpaqueTriangleHit(); // since we save the min, we can commit here
                        }
                    }
                }
                
                // calculate visibility inside and outside of sphere
                float sphereZ = posVLength - tFirstFrontFaceInside * posVLength / initialSamplePosLength;
                curVisibility = calcSphereVisibility(sphereZ, sphereStart, sphereEnd, pdf);

                if (occlusionStack > 0)
                    curVisibility = 0.0;
            }
            
        }

        visibility += curVisibility;
        
        // advance mask for next iteration
        mask = mask >> 1;
        ++i;
    }

    visibility *= 2.0 / float(NUM_DIRECTIONS);
    float prevAo = aoPrev.SampleLevel(gTextureSampler, texC, 0);
    visibility += prevAo;
    visibility = pow(visibility, gData.exponent);
    return visibility;
}
