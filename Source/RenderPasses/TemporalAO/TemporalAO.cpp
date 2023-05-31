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
#include "TemporalAO.h"
#include "../Utils/GuardBand/guardband.h"

namespace
{
    const std::string kMotionVec = "mvec";
    const std::string kAOIn = "aoIn";
    const std::string kAOOut = "aoOut";
    const std::string kDepth = "linearZ"; // linear?

    const std::string kShaderFilename = "RenderPasses/TemporalAO/TemporalAO.ps.slang";
    
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, TemporalAO>();
}

TemporalAO::TemporalAO(ref<Device> pDevice, const Dictionary& dict)
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

    { // ao sampler
        Sampler::Desc samplerDesc;
        samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
        samplerDesc.setAddressingMode(Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp);
        mpPass->getRootVar()["gAOSampler"] = Sampler::create(pDevice, samplerDesc);
    }
    
    
    // TODO set sampler
}

Dictionary TemporalAO::getScriptingDictionary()
{
    return Dictionary();
}

RenderPassReflection TemporalAO::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;
    reflector.addInput(kAOIn, "AO").format(ResourceFormat::R8Unorm).bindFlags(Resource::BindFlags::ShaderResource);
    reflector.addInput(kDepth, "linear? depths").bindFlags(Resource::BindFlags::ShaderResource);
    reflector.addInput(kMotionVec, "Motion vectors").bindFlags(Resource::BindFlags::ShaderResource);
    reflector.addOutput(kAOOut, "AO").format(ResourceFormat::R8Unorm).bindFlags(ResourceBindFlags::AllColorViews);
    return reflector;
}

void TemporalAO::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpScene) return;

    auto pAOIn = renderData[kAOIn]->asTexture();
    auto pDepth = renderData[kDepth]->asTexture();
    auto pMotionVec = renderData[kMotionVec]->asTexture();
    auto pAOOut = renderData[kAOOut]->asTexture();

    if (!mEnabled)
    {
        pRenderContext->blit(pAOIn->getSRV(), pAOOut->getRTV());
        return;
    }

    // check if resource dimensions changed and allocate texture accordingly
    mpPrevAO = allocatePrevFrameTexture(pAOOut, std::move(mpPrevAO));
    mpPrevDepth = allocatePrevFrameTexture(pDepth, std::move(mpPrevDepth));

    auto& dict = renderData.getDictionary();
    auto guardBand = dict.getValue("guardBand", 0);
    setGuardBandScissors(*mpPass->getState(), renderData.getDefaultTextureDims(), guardBand);

    auto vars = mpPass->getRootVar();
    vars["gMotionVec"] = pMotionVec;
    vars["gDepth"] = pDepth;
    vars["gPrevDepth"] = mpPrevDepth;
    vars["gAO"] = pAOIn;
    vars["gPrevAO"] = mpPrevAO;
    mpScene->getCamera()->setShaderData(vars["PerFrameCB"]["gCamera"]);
    auto conversionMat = math::mul(mpScene->getCamera()->getViewMatrix(), math::inverse(mpScene->getCamera()->getPrevViewMatrix()));
    vars["PerFrameCB"]["prevViewToCurView"] = conversionMat;
    vars["PerFrameCB"]["uvMin"] = dict.getValue("guardBand.uvMin", float2(0.0f));
    vars["PerFrameCB"]["uvMax"] = dict.getValue("guardBand.uvMax", float2(1.0f));

    mpFbo->attachColorTarget(pAOOut, 0);
    mpPass->execute(pRenderContext, mpFbo, false);

    // save depth and ao from this frame for next frame
    pRenderContext->blit(pDepth->getSRV(), mpPrevDepth->getRTV());
    pRenderContext->blit(pAOOut->getSRV(), mpPrevAO->getRTV());
}

void TemporalAO::renderUI(Gui::Widgets& widget)
{
    widget.checkbox("Enable", mEnabled);
}

void TemporalAO::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
}

ref<Texture> TemporalAO::allocatePrevFrameTexture(const ref<Texture>& original, ref<Texture> prev) const
{
    assert(original);
    bool allocate = prev == nullptr;
    allocate = allocate || (prev->getWidth() != original->getWidth());
    allocate = allocate || (prev->getHeight() != original->getHeight());
    allocate = allocate || (prev->getFormat() != original->getFormat());
    
    if (!allocate) return prev;

    return Texture::create2D(mpDevice, original->getWidth(), original->getHeight(), original->getFormat(), 1, 1, nullptr, original->getBindFlags());
}
