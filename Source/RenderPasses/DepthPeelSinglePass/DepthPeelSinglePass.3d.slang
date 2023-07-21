/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
import Scene.Raster;
import Utils.Math.MathHelpers;

struct VSOutEx
{
    VSOut vsOut;
    uint rtIndex : SV_RenderTargetArrayIndex;
};

VSOutEx vsMain(VSIn vIn)
{
    VSOutEx o;
    o.vsOut = defaultVS(vIn);
    o.rtIndex = 1;
    return o;
}

/*struct GSIn
{
    VSOut vsOut;
    //uint triangleIndex : SV_PrimitiveID;
};

struct GSOut
{
    VSOut vsOut;
    uint rtIndex : SV_RenderTargetArrayIndex;
    uint triangleIndex : SV_PrimitiveID;
};

[maxvertexcount(6)]
void gsMain(triangle GSIn gin[3], uint triangleIndex : SV_PrimitiveID, inout TriangleStream<GSOut> t)
{
    GSOut o;
    o.triangleIndex = triangleIndex;
    
    [unroll]
    for (uint rtIndex = 0; rtIndex < 2; ++rtIndex)
    {
        o.rtIndex = rtIndex;
        
        o.vsOut = gin[0].vsOut;
        t.Append(o);
        o.vsOut = gin[1].vsOut;
        t.Append(o);
        o.vsOut = gin[2].vsOut;
        t.Append(o);
        t.RestartStrip();
    }
}*/

Texture2D<float> prevDepthTex;

cbuffer PerFrameCB
{
    uint2 gFrameDim;
};

float2 computeMotionVector(const VSOut vsOut, const int2 ipos)
{
    float2 pixelPos = ipos + float2(0.5, 0.5); // Current sample in pixel coords.
    float4 prevPosH = vsOut.prevPosH; // Sample in previous frame in clip space coords, no jittering applied.
    return calcMotionVector(pixelPos, prevPosH, gFrameDim) + float2(gScene.camera.data.jitterX, -gScene.camera.data.jitterY); // Remove camera jitter from motion vector
}

void psMain(VSOut vsOut, uint rtIndex : SV_RenderTargetArrayIndex, uint triangleIndex : SV_PrimitiveID) : SV_TARGET
{
#if USE_ALPHA_TEST
    //let lod = ImplicitLodTextureSampler();
    let lod = ExplicitLodTextureSampler(0.0); // same as raytracer
    if (alphaTest(vsOut, triangleIndex, lod)) discard;
#endif
    
    if (rtIndex == 0)
        return; // return for normal depth buffer

    // do approximate depth peeling
    //float2 mvec = computeMotionVector(vsOut, int2(vsOut.posH.xy));

    /*float prevDepth = prevDepthTex[int2(vsOut.posH.xy)].x;
    if (vsOut.posH.z <= prevDepth)
        discard; // only accept smaller depths
*/
}
