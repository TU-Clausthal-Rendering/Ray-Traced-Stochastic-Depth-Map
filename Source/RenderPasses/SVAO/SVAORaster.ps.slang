import Scene.RaytracingInline;
#include "Common.slang"

#if SECONDARY_DEPTH_MODE == DEPTH_MODE_SINGLE
// disable all stencil operations if no secondary pass exists
#define STENCIL(x) 
#else
#define STENCIL(x) x
#endif

struct PSOut
{
    ao_t ao1 : SV_TARGET0;
    STENCIL(uint stencil : SV_TARGET1);
};

RWTexture2D<uint> gRayMinAccess;
RWTexture2D<uint> gRayMaxAccess;

PSOut main(float2 texC : TEXCOORD, float4 svPos : SV_POSITION)
{
    PSOut output;
    output.ao1 = 0.0;
    STENCIL(output.stencil = 0);

    BasicAOData data;
    if (!data.Init(texC))
    {
        output.ao1 = 1.0;
        return output;
    }

    float visibility = 0.0f;    
    [unroll] for (uint i = 0; i < NUM_DIRECTIONS; i++)
    {
        SampleAOData s;
        if (!s.Init(texC, data, i))
            continue;

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
        if (!data.denyRays() && (s.requireRay(data) || forceRay))
        {
            // remember that this was a hybrid ray
            STENCIL(output.stencil |= 1u << i);
            
#if (SECONDARY_DEPTH_MODE == DEPTH_MODE_STOCHASTIC)
            #if USE_RAY_INTERVAL
            InterlockedMin(gRayMinAccess[uint2(s.rasterSamplePosUV * gData.resolution) / STOCH_MAP_DIVISOR], /*rough estimate of depth*/asuint(max(data.posVLength - data.radius - gData.thickness * data.radius - s.sphereStart, 0.0)));
            InterlockedMax(gRayMaxAccess[uint2(s.rasterSamplePosUV * gData.resolution) / STOCH_MAP_DIVISOR], /*rough estimate of depth*/asuint(max(data.posVLength - s.sphereEnd, 0.0)));
            #else // no interval (only set flag)
            gRayMaxAccess[uint2(s.rasterSamplePosUV * gData.resolution) / STOCH_MAP_DIVISOR] = 1u;
            #endif          
#endif
        }
        else
        {
            darkmap(output.ao1.y += s.visibility); // also add on dark if no ray is required (reference solution)
        }
    }

    output.ao1 *= 2.0 / float(NUM_DIRECTIONS);

    #if SECONDARY_DEPTH_MODE == DEPTH_MODE_SINGLE
    output.ao1 = pow(output.ao1, gData.exponent); // there will be no second shader
    #else
    // apply exponent
    if(output.stencil == 0)
        output.ao1 = pow(output.ao1, gData.exponent);
    #endif
    
    return output;
}
