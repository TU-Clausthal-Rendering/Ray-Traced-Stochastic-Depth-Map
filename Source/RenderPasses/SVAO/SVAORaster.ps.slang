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
                if(AO_KERNEL == AO_KERNEL_HBAO) output.ao1 += 0.0;
                if(AO_KERNEL == AO_KERNEL_VAO) output.ao1 += (s.sphereStart - s.sphereEnd) / s.pdf;
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

            if (!s.isInScreen && gData.sdGuard > 0)
            {
                forceRay = true; // always shoot rays for screen border
                s.objectSpaceZ = 3.402823466e+38F; // reset object space for new ray interval calculation
            }
            
            if (s.requireRay(data) || forceRay)
            {
                STENCIL(output.stencil |= 1u << i);
            
#if (SECONDARY_DEPTH_MODE == DEPTH_MODE_STOCHASTIC)
                uint2 pixel = UVToSDPixel(s.samplePosUV);
#if USE_RAY_INTERVAL
                uint rayMin, rayMax;
                float objectSpaceMin;
                if(AO_KERNEL == AO_KERNEL_VAO) objectSpaceMin = min(s.objectSpaceZ, data.radius + gData.thickness * data.radius + s.sphereStart);
                if(AO_KERNEL == AO_KERNEL_HBAO) objectSpaceMin = min(s.objectSpaceZ, s.sphereStart);
                rayMin = /*rough estimate of depth*/asuint(max(data.posVLength - objectSpaceMin, 0.0));
                rayMax = /*rough estimate of depth*/asuint(max(data.posVLength - s.sphereEnd, 0.0));
                InterlockedMin(gRayMinAccess[pixel], rayMin);
                InterlockedMax(gRayMaxAccess[pixel], rayMax);
#else // no interval (only set flag)
                gRayMaxAccess[pixel] = 1u;
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
        output.ao1 = data.finalize(output.ao1); // there is no second shader
#else
        // apply exponent
        if (output.stencil == 0)
        {
            output.ao1 = data.finalize(output.ao1);
        }
#endif
    }
    
    writeOutput(svPos, output);
}
