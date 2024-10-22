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
#include "CrossBilateralBlur.h"
#include "../Utils/GuardBand/guardband.h"

namespace
{
    const char kShaderPath[] = "RenderPasses/CrossBilateralBlur/CrossBilateralBlur.ps.slang";

    const std::string kColor = "color";
    const std::string kColorOut = "colorOut";
    const std::string kDepth = "linear depth";
    const std::string kPingPong = "pingpong";
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, CrossBilateralBlur>();
}

CrossBilateralBlur::CrossBilateralBlur(ref<Device> pDevice) : RenderPass(std::move(pDevice))
{
    mpFbo = Fbo::create(mpDevice);
    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(Sampler::Filter::Point, Sampler::Filter::Point, Sampler::Filter::Point).setAddressingMode(Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp);
    mpSampler = Sampler::create(mpDevice, samplerDesc);
}

ref<CrossBilateralBlur> CrossBilateralBlur::create(ref<Device> pDevice, const Properties& dict)
{
    auto pPass = make_ref<CrossBilateralBlur>(std::move(pDevice));
    for (const auto& [key, value] : dict)
    {
        logWarning("Unknown field '" + key + "' in a CrossBilateralBlur dictionary");
    }
    return pPass;
}

Properties CrossBilateralBlur::getProperties() const
{
    Properties dict;
    return dict;
}

RenderPassReflection CrossBilateralBlur::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;
    mReady = false;

    reflector.addInput(kColor, "color image to be blurred").bindFlags(ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget);
    reflector.addInput(kDepth, "linear depth").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addInternal(kPingPong, "temporal result after first blur").format(mFormat).bindFlags(ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget);
    reflector.addOutput(kColorOut, "blurred color").format(mFormat).bindFlags(ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget);

    // set correct input format and dimensions of the ping pong buffer
    auto edge = compileData.connectedResources.getField(kColor);
    if (edge)
    {
        const auto srcWidth = edge->getWidth();
        const auto srcHeight = edge->getHeight();

        reflector.addInternal(kPingPong, "").texture2D(srcWidth, srcHeight, 1, 1, 1);
        reflector.addOutput(kColorOut, "").texture2D(srcWidth, srcHeight, 1, 1, 1);

        mReady = true;
    }

    return reflector;
}

void CrossBilateralBlur::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    if(!mReady) throw std::runtime_error("CrossBilateralBlur::compile - missing incoming reflection information");

    DefineList defines;
    defines.add("KERNEL_RADIUS", std::to_string(mKernelRadius));

    mpBlur = FullScreenPass::create(mpDevice, kShaderPath, defines);

    mpBlur->getRootVar()["gSampler"] = mpSampler;
}

void CrossBilateralBlur::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto pColor = renderData[kColor]->asTexture();
    auto pColorOut = renderData[kColorOut]->asTexture();
    auto pPingPong = renderData[kPingPong]->asTexture();
    auto pDepth = renderData[kDepth]->asTexture();

    if (!mEnabled)
    {
        pRenderContext->blit(pColor->getSRV(), pColorOut->getRTV());
        return;
    }

    // set resources if they changed
    if (mpBlur->getRootVar()["gDepthTex"].getTexture() != pDepth)
    {
        mpBlur->getRootVar()["gDepthTex"] = pDepth;
    }

    auto& dict = renderData.getDictionary();
    auto guardBand = dict.getValue("guardBand", 0);
    setGuardBandScissors(*mpBlur->getState(), renderData.getDefaultTextureDims(), guardBand);
    // set scissor cb (is shared between both shaders)
    mpBlur->getRootVar()["ScissorCB"]["uvMin"] = dict.getValue("guardBand.uvMin", float2(0.0f));
    mpBlur->getRootVar()["ScissorCB"]["uvMax"] = dict.getValue("guardBand.uvMax", float2(1.0f));
    mpBlur->getRootVar()["CB"]["betterSlope"] = (mBetterSlope ? 1 : 0);

    for (uint32_t i = 0; i < mRepetitions; ++i)
    {
        // blur in x
        mpBlur->getRootVar()["gSrcTex"] = pColor;
        mpFbo->attachColorTarget(pPingPong, 0);
        mpBlur->getRootVar()["CB"]["dir"] = float2(1.0f, 0.0f);
        mpBlur->execute(pRenderContext, mpFbo, false);

        // blur in y
        mpBlur->getRootVar()["gSrcTex"] = pPingPong;
        mpFbo->attachColorTarget(pColorOut, 0);
        mpBlur->getRootVar()["CB"]["dir"] = float2(0.0f, 1.0f);
        mpBlur->execute(pRenderContext, mpFbo, false);
    }
}

void CrossBilateralBlur::renderUI(Gui::Widgets& widget)
{
    widget.checkbox("Enabled", mEnabled);
    if (!mEnabled) return;

    if (widget.var("Kernel Radius", mKernelRadius, uint32_t(1), uint32_t(20))) requestRecompile();
    widget.checkbox("Better Slope", mBetterSlope);
    widget.var("Blur Repetitions", mRepetitions, uint32_t(1), uint32_t(20));
}
