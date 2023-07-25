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
#include "TemporalDepthPeel.h"
//#include "../Utils/GuardBand/guardband.h"

namespace
{
    const std::string kMotionVec = "mvec";
    const std::string kDepth = "linearZ";
    const std::string kDepthOut = "depth2";

    const std::string kShaderFilename = "RenderPasses/TemporalDepthPeel/TemporalDepthPeel.ps.slang";

}
extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, TemporalDepthPeel>();
}

TemporalDepthPeel::TemporalDepthPeel(ref<Device> pDevice, const Properties& props)
    : RenderPass(pDevice)
{
    mpPass = FullScreenPass::create(mpDevice, kShaderFilename);
    mpFbo = Fbo::create(pDevice);

    { // depth sampler
        Sampler::Desc samplerDesc;
        samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
        samplerDesc.setAddressingMode(Sampler::AddressMode::Border, Sampler::AddressMode::Border, Sampler::AddressMode::Border);
        samplerDesc.setBorderColor(float4(0.0f));
        mpPass->getRootVar()["gDepthSampler"] = Sampler::create(pDevice, samplerDesc);
    }

}

Properties TemporalDepthPeel::getProperties() const
{
    return {};
}

RenderPassReflection TemporalDepthPeel::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addInput(kDepth, "linear depths").bindFlags(Resource::BindFlags::ShaderResource);
    reflector.addInput(kMotionVec, "Motion vectors").bindFlags(Resource::BindFlags::ShaderResource);
    reflector.addOutput(kDepthOut, "depthOut").format(ResourceFormat::R32Float).bindFlags(ResourceBindFlags::AllColorViews);
    return reflector;
}

void TemporalDepthPeel::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    mpPrevDepth.reset();
    mpPrevDepth2.reset();
}

void TemporalDepthPeel::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpScene) return;

    auto pDepth = renderData[kDepth]->asTexture();
    auto pMotionVec = renderData[kMotionVec]->asTexture();
    auto pDepthOut = renderData[kDepthOut]->asTexture();

    if (!mEnabled)
    {
        pRenderContext->blit(pDepth->getSRV(), pDepthOut->getRTV());
        mpPrevDepth.reset();
        mpPrevDepth2.reset();
        return;
    }

    // check if resource dimensions changed and allocate texture accordingly
    mpPrevDepth = allocatePrevFrameTexture(pDepth, std::move(mpPrevDepth));
    mpPrevDepth2 = allocatePrevFrameTexture(pDepth, std::move(mpPrevDepth2));

    //auto& dict = renderData.getDictionary();
    //auto guardBand = dict.getValue("guardBand", 0);
    //setGuardBandScissors(*mpPass->getState(), renderData.getDefaultTextureDims(), guardBand);

    auto vars = mpPass->getRootVar();
    vars["gMotionVec"] = pMotionVec;
    vars["gDepth"] = pDepth;
    vars["gPrevDepth"] = mpPrevDepth;
    vars["gPrevDepth2"] = mpPrevDepth2;

    mpScene->getCamera()->setShaderData(vars["PerFrameCB"]["gCamera"]);
    auto conversionMat = math::mul(mpScene->getCamera()->getViewMatrix(), math::inverse(mpScene->getCamera()->getPrevViewMatrix()));
    vars["PerFrameCB"]["prevViewToCurView"] = conversionMat;
    conversionMat = math::mul(mpScene->getCamera()->getPrevViewMatrix(), math::inverse(mpScene->getCamera()->getViewMatrix()));
    vars["PerFrameCB"]["curViewToPrevView"] = conversionMat;
    //vars["PerFrameCB"]["uvMin"] = dict.getValue("guardBand.uvMin", float2(0.0f));
    //vars["PerFrameCB"]["uvMax"] = dict.getValue("guardBand.uvMax", float2(1.0f));

    mpFbo->attachColorTarget(pDepthOut, 0);
    mpPass->execute(pRenderContext, mpFbo, true);

    // save depth and ao from this frame for next frame
    pRenderContext->blit(pDepth->getSRV(), mpPrevDepth->getRTV());
    pRenderContext->blit(pDepthOut->getSRV(), mpPrevDepth2->getRTV());
}

void TemporalDepthPeel::renderUI(Gui::Widgets& widget)
{
    widget.checkbox("Enable", mEnabled);
}

void TemporalDepthPeel::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
}

ref<Texture> TemporalDepthPeel::allocatePrevFrameTexture(const ref<Texture>& original, ref<Texture> prev) const
{
    assert(original);
    bool allocate = prev == nullptr;
    allocate = allocate || (prev->getWidth() != original->getWidth());
    allocate = allocate || (prev->getHeight() != original->getHeight());
    allocate = allocate || (prev->getFormat() != original->getFormat());

    if (!allocate) return prev;

    return Texture::create2D(mpDevice, original->getWidth(), original->getHeight(), original->getFormat(), 1, 1, nullptr, original->getBindFlags());
}
