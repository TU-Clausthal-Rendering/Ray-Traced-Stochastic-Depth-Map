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
RWTexture2D<uint> gRayMinAccess;
RWTexture2D<uint> gRayMaxAccess;

PSOut main(float2 texC : TEXCOORD, float4 svPos : SV_POSITION)
{
    PSOut output;
    output.ao1 = 0.0;
    output.stencil = 0;

    BasicAOData data;
    if (!data.Init(texC))
    {
        output.ao1 = 1.0;
        return output;
    }

    float3 color = saturate(gColor.SampleLevel(gTextureSampler, texC, 0).xyz); // for pixel importance
    
    float visibility = 0.0f;

    uint forceRayMask = 0; // forces rays due to double sided materials
    
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
            float zIntersect = -dot(dir.xy, data.normalO.xy) / makeNonZero(data.normalO.z, 0.0001);
            float zIntersectClamped = clamp(zIntersect, -sphereHeight, sphereHeight);
            sphereEnd = zIntersectClamped;
        }
        
        // sample position calculate uv position of sample
        float3 initialSamplePosV = data.posV + data.tangent * dir.x + data.bitangent * dir.y;
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
        
        float objectSpaceZ = calcObjectSpaceZ(data.posV, data.normal, rasterSamplePosUV, gDepthTex);
        curVisibility = calcVisibility(objectSpaceZ, sphereStart, sphereEnd, pdf);

        if ((sphereStart - sphereEnd) / pdf <= 0.1)
        {
            continue; // skip sample (no visibility)
        }

        float TRUSTED_AREA = sphereStart + CONST_RADIUS;
        
        // require hybrid ray if intersection is outside of sphere radius
        if (PRIMARY_DEPTH_MODE != DEPTH_MODE_DUAL && objectSpaceZ > TRUSTED_AREA)
        {
            requireRay = true;
        }

        if (PRIMARY_DEPTH_MODE == DEPTH_MODE_DUAL && objectSpaceZ > TRUSTED_AREA)
        {
            float objectSpaceZ = calcObjectSpaceZ(data.posV, data.normal, rasterSamplePosUV, gDepthTex2);
            curVisibility = COMBINE_VIS(curVisibility, calcVisibility(objectSpaceZ, sphereStart, sphereEnd, pdf));
            if (objectSpaceZ > TRUSTED_AREA) // both samples were before the sphere starts
                requireRay = true;
        }

        // always add computed visibility from raster
        output.ao1 += curVisibility;
        if (data.denyRays())
            continue;

        // calc pixel importance
        float sampleImportance = calcSampleImportance(radius, gDepthTex.SampleLevel(gTextureSampler, rasterSamplePosUV, 0.0), sphereStart, sphereEnd, objectSpaceZ, color);
        if (sampleImportance <= gData.importanceThreshold)
            continue;
        
        if (requireRay || forceRay)
        {
            // remember that this was a hybrid ray
            output.stencil |= 1u << i;
            
#if (SECONDARY_DEPTH_MODE == DEPTH_MODE_STOCHASTIC)
            gDepthAccess[uint2(rasterSamplePosUV * gData.resolution)] = 1u;
            InterlockedMin(gRayMinAccess[uint2(rasterSamplePosUV * gData.resolution)], /*rough estimate of depth*/asuint(max(data.posVLength - sphereStart - CONST_RADIUS - HALO_RADIUS, 0.0)));
            InterlockedMax(gRayMaxAccess[uint2(rasterSamplePosUV * gData.resolution)], /*rough estimate of depth*/asuint(max(data.posVLength - sphereEnd, 0.0)));
#endif
        }
    }

    output.ao1 *= 2.0 / float(NUM_DIRECTIONS);

    // apply exponent
    if(output.stencil == 0)
        output.ao1 = pow(output.ao1, gData.exponent);
    
    return output;
}
