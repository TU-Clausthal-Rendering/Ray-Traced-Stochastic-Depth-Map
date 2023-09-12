/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#include "ZMipmaps.h"

namespace
{
    const std::string kDepth = "linearZ";

    const std::string kDepthOut = "linearZMip";
    const std::string kMipLevels = "mipLevels";
    const std::string kThreshold = "threshold";

    const std::string kMipShader = "RenderPasses/ZMipmaps/Mip.ps.slang";
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, ZMipmaps>();
}

ZMipmaps::ZMipmaps(ref<Device> pDevice, const Properties& props)
    : RenderPass(pDevice)
{
    for (const auto& [key, value] : props)
    {
        if (key == kMipLevels) mMipLevels = value;
        else if(key == kThreshold) mThreshold = value;
        else logWarning("Unknown field `" + key + "` in a ZMipmaps dictionary");
    }

    mpFbo = Fbo::create(pDevice);

    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(Sampler::Filter::Point, Sampler::Filter::Point, Sampler::Filter::Point);
    auto pointSampler = Sampler::create(pDevice, samplerDesc);

    mpMipPass = FullScreenPass::create(pDevice, kMipShader);
    mpMipPass->getRootVar()["S"] = pointSampler;
}

Properties ZMipmaps::getProperties() const
{
    Properties dict;
    dict[kMipLevels] = mMipLevels;
    dict[kThreshold] = mThreshold;
    return dict;
}

RenderPassReflection ZMipmaps::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addInput(kDepth, "linear depth buffer").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addOutput(kDepthOut, "linear depth mipmaps").bindFlags(ResourceBindFlags::AllColorViews).format(ResourceFormat::R32Float)
        .texture2D(0, 0, 1, mMipLevels, 1);

    return reflector;
}

void ZMipmaps::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto pDepthIn = renderData[kDepth]->asTexture();
    auto pDepthOut = renderData[kDepthOut]->asTexture();

    float farZ = 1e10f;
    if(mpScene)
    {
        farZ = mpScene->getCamera()->getFarPlane();
    }

    auto vars = mpMipPass->getRootVar();
    vars["CameraCB"]["farZ"] = farZ;


    // copy first level
    pRenderContext->blit(pDepthIn->getSRV(0, 1), pDepthOut->getRTV());

    // gen mipmaps
    for(uint i = 1;  i < pDepthOut->getMipCount(); i++)
    {
        // adaptive threshold based on mip level
        float t = mThreshold;
        if(mAdaptiveThreshold)
            t = t / (t + powf(2, float(i-1)) * (1 - t));

        mpFbo->attachColorTarget(pDepthOut, 0, i);
        vars["gSrc"].setSrv(pDepthOut->getSRV(i - 1, 1));
        vars["CameraCB"]["threshold"] = t;
        mpMipPass->execute(pRenderContext, mpFbo);
    }
}

void ZMipmaps::renderUI(Gui::Widgets& widget)
{
    if (widget.var("Mip Levels", mMipLevels, -1, 14))
        requestRecompile();
    widget.var("Threshold", mThreshold, 0.f, 1.f, 0.01f);
    widget.checkbox("Adaptive Threshold", mAdaptiveThreshold);
}
