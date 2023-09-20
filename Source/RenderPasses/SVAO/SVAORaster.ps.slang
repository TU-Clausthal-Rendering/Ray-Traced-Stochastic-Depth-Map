import Scene.RaytracingInline;
#include "Common.slang"

struct PSOut
{
    ao_t ao1 : SV_TARGET0;
    uint stencil : SV_TARGET1;
};

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

    #if USE_IMPORTANCE
    float3 color = saturate(gColor.SampleLevel(gTextureSampler, texC, 0).xyz); // for pixel importance
    #endif

    float visibility = 0.0f;    
    [unroll] for (uint i = 0; i < NUM_DIRECTIONS; i++)
    {
        SampleAOData s;
        if (!s.Init(texC, data, i))
            continue;

        uint skipId = (uint(svPos.x) % pixelSkipX) + (uint(svPos.y) % pixelSkipY) * pixelSkipX;
        if (skipId != (frameIndex % (pixelSkipX * pixelSkipY)))
            data.forceDenyRays();

        if (isSamePixel(texC, s.rasterSamplePosUV))
        {
            output.ao1 += (s.sphereStart - s.sphereEnd) / s.pdf;
            continue;
        }

        bool forceRay = false;
        if (!s.isInScreen && SECONDARY_DEPTH_MODE == DEPTH_MODE_RAYTRACING && TRACE_OUT_OF_SCREEN)
        {
            forceRay = true; // always shoot rays for screen border   
        }
        
        s.evalPrimaryVisibility(data);
        if (PRIMARY_DEPTH_MODE == DEPTH_MODE_DUAL)
            s.evalDualVisibility(data);

        // always add computed visibility from raster
        output.ao1.x += s.visibility; // only on bright       
        if (!data.denyRays() && (s.requireRay || forceRay))
        {
            // calc pixel importance
            #if USE_IMPORTANCE
            float sampleImportance = calcSampleImportance(sampleRadius[i] * gData.radius, gDepthTex.SampleLevel(gTextureSampler, s.rasterSamplePosUV, 0.0), s.sphereStart, s.sphereEnd, s.objectSpaceZ, color);
            if (sampleImportance <= gData.importanceThreshold) continue;
            #endif

            // remember that this was a hybrid ray
            output.stencil |= 1u << i;
            
#if (SECONDARY_DEPTH_MODE == DEPTH_MODE_STOCHASTIC)
            InterlockedMin(gRayMinAccess[uint2(s.rasterSamplePosUV * gData.resolution) / STOCH_MAP_DIVISOR], /*rough estimate of depth*/asuint(max(data.posVLength - gData.radius - gData.thickness * gData.radius - s.sphereStart, 0.0)));
            InterlockedMax(gRayMaxAccess[uint2(s.rasterSamplePosUV * gData.resolution) / STOCH_MAP_DIVISOR], /*rough estimate of depth*/asuint(max(data.posVLength - s.sphereEnd, 0.0)));
            //InterlockedMin(gRayMinAccess[uint2(s.rasterSamplePosUV * gData.resolution)], /*rough estimate of depth*/asuint(max(data.linearDepth - gData.radius - gData.thickness * gData.radius - s.sphereStart, 0.0)));
            //InterlockedMax(gRayMaxAccess[uint2(s.rasterSamplePosUV * gData.resolution)], /*rough estimate of depth*/asuint(max(data.linearDepth - s.sphereEnd, 0.0)));
#endif
        }
        else
        {
            darkmap(output.ao1.y += s.visibility); // also add on dark if no ray is required (reference solution)
        }
    }

    output.ao1 *= 2.0 / float(NUM_DIRECTIONS);

    // apply exponent
    if(output.stencil == 0)
        output.ao1 = pow(output.ao1, gData.exponent);
    
    return output;
}
