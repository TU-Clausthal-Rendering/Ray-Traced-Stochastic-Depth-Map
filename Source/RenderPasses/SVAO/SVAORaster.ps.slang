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
    ao_t ao1;
    STENCIL(uint stencil);
};

RWTexture2D<uint> gRayMinAccess;
RWTexture2D<uint> gRayMaxAccess;

RWTexture2D<unorm float> gAO1;
RWTexture2D<uint> gStencil;

void writeOutput(uint2 pixel, PSOut output)
{
    gAO1[pixel] = output.ao1;
    STENCIL(gStencil[pixel] = output.stencil);
}

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID, uint3 localId : SV_GroupThreadID)
{
    PSOut output;
    output.ao1 = 0.0;
    STENCIL(output.stencil = 0);

    uint2 offset = (group.xy / 2u) * 32u + 2u * localId.xy + (group.xy % 2u); // 2x2 group alignment works better because of the per-pixel rotation. See SVAO::genNoiseTexture()
    uint2 svPos = offset + uint2(guardBand); // pixel position
    float2 texC = (svPos + float2(0.5)) * gData.invResolution; // texture position

    BasicAOData data;
    if (!data.Init(texC))
    {
        output.ao1 = 1.0;
    }
    else
    {
        float visibility = 0.0;
        [unroll]
        for (uint i = 0; i < NUM_DIRECTIONS; i++)
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
        
            if (s.requireRay(data) || forceRay)
            {
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

        output.ao1 *= 1.0 / float(NUM_DIRECTIONS);
        if(AO_KERNEL == AO_KERNEL_VAO) output.ao1 *= 2.0;

#if SECONDARY_DEPTH_MODE == DEPTH_MODE_SINGLE
        if(AO_KERNEL == AO_KERNEL_HBAO) output.ao1 = saturate(1.0 - 2.0 * output.ao1);
        output.ao1 = pow(output.ao1, gData.exponent); // there will be no second shader
#else
        // apply exponent
        if (output.stencil == 0)
        {
            if(AO_KERNEL == AO_KERNEL_HBAO) output.ao1 = saturate(1.0 - 2.0 * output.ao1);
            output.ao1 = pow(output.ao1, gData.exponent);
        }
#endif
    }
    
    writeOutput(svPos, output);
}
