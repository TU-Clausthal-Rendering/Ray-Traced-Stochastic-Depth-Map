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
        output.ao1 += s.visibility;
        if (data.denyRays())
            continue;

        // calc pixel importance
        float sampleImportance = calcSampleImportance(sampleRadius[i] * gData.radius, gDepthTex.SampleLevel(gTextureSampler, s.rasterSamplePosUV, 0.0), s.sphereStart, s.sphereEnd, s.objectSpaceZ, color);
        if (sampleImportance <= gData.importanceThreshold)
            continue;
        
        if (s.requireRay || forceRay)
        {
            // remember that this was a hybrid ray
            output.stencil |= 1u << i;
            
#if (SECONDARY_DEPTH_MODE == DEPTH_MODE_STOCHASTIC)
            gDepthAccess[uint2(s.rasterSamplePosUV * gData.resolution)] = 1u;
            InterlockedMin(gRayMinAccess[uint2(s.rasterSamplePosUV * gData.resolution)], /*rough estimate of depth*/asuint(max(data.posVLength - gData.radius - gData.thickness * gData.radius - s.sphereStart, 0.0)));
            InterlockedMax(gRayMaxAccess[uint2(s.rasterSamplePosUV * gData.resolution)], /*rough estimate of depth*/asuint(max(data.posVLength - s.sphereEnd, 0.0)));
#endif
        }
    }

    output.ao1 *= 2.0 / float(NUM_DIRECTIONS);

    // apply exponent
    if(output.stencil == 0)
        output.ao1 = pow(output.ao1, gData.exponent);
    
    return output;
}
