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
#include "AOGuidedBlur.h"
#include "../Utils/GuardBand/guardband.h"

namespace
{
    const char kShaderPath[] = "RenderPasses/AOGuidedBlur/AOGuidedBlur.ps.slang";

    const std::string kBright = "bright";
    const std::string kDark = "dark";
    //const std::string kImportance = "importance";
    const std::string kDepth = "lineardepth";
    const std::string kPingPong = "pingpong";

    const std::string kOutput = "color";
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, AOGuidedBlur>();
}

ref<AOGuidedBlur> AOGuidedBlur::create(ref<Device> pDevice, const Dictionary& dict)
{
    auto pPass = make_ref<AOGuidedBlur>(pDevice, dict);
    for (const auto& [key, value] : dict)
    {
        logWarning("Unknown field '" + key + "' in a AOGuidedBlur dictionary");
    }
    return pPass;
}

AOGuidedBlur::AOGuidedBlur(ref<Device> pDevice, const Dictionary& dict)
    : RenderPass(pDevice)
{
    mpFbo = Fbo::create(mpDevice);
    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(Sampler::Filter::Point, Sampler::Filter::Point, Sampler::Filter::Point).setAddressingMode(Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp);
    mpSampler = Sampler::create(mpDevice, samplerDesc);
}

Dictionary AOGuidedBlur::getScriptingDictionary()
{
    Dictionary dict;
    return dict;
}

RenderPassReflection AOGuidedBlur::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    mReady = false;
    
    reflector.addInput(kBright, "bright ao (will be overwritten)").bindFlags(ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget).texture2D(0,0,1,1,0);
    reflector.addInput(kDark, "dark ao").bindFlags(ResourceBindFlags::ShaderResource).texture2D(0, 0, 1, 1, 0);
    reflector.addInput(kDepth, "linear depth").bindFlags(ResourceBindFlags::ShaderResource).texture2D(0,0,1,1,0);
    //reflector.addInput(kImportance, "importance map").bindFlags(ResourceBindFlags::ShaderResource).texture2D(0,0,1,1,0);
    reflector.addInternal(kPingPong, "temporal result after first blur").bindFlags(ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget);
    reflector.addOutput(kOutput, "blurred ao").bindFlags(ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget);

    // set correct input format and dimensions of the ping pong buffer
    auto edge = compileData.connectedResources.getField(kBright);
    if (edge)
    {
        const auto inputFormat = edge->getFormat();
        const auto srcWidth = edge->getWidth();
        const auto srcHeight = edge->getHeight();
        const auto srcArraySize = edge->getArraySize();

        mLastFormat = inputFormat;
        reflector.addInternal(kPingPong, "").format(inputFormat).texture2D(srcWidth, srcHeight, 1, 1, srcArraySize);
        reflector.addOutput(kOutput, "").format(inputFormat).texture2D(srcWidth, srcHeight, 1, 1, srcArraySize);

        mReady = true;
    }

    return reflector;
}

void AOGuidedBlur::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    if (!mReady) throw std::runtime_error("CrossBilateralBlur::compile - missing incoming reflection information");

    Program::DefineList defines;
    defines.add("KERNEL_RADIUS", std::to_string(mKernelRadius));

    mpBlur = FullScreenPass::create(mpDevice, kShaderPath, defines);

    mpBlur->getRootVar()["gSampler"] = mpSampler;
}

void AOGuidedBlur::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto pBright = renderData[kBright]->asTexture();
    auto pDark = renderData[kDark]->asTexture();
    auto pPingPong = renderData[kPingPong]->asTexture();
    auto pDepth = renderData[kDepth]->asTexture();
    //auto pImportance = renderData[kImportance]->asTexture();

    auto pOutput = renderData[kOutput]->asTexture();

    assert(pBright->getFormat() == pPingPong->getFormat());
    assert(pBright->getFormat() == pOutput->getFormat());

    if (!mEnabled)
    {
        // blit input to output
        for (uint slice = 0; slice < pBright->getArraySize(); ++slice)
        {
            pRenderContext->blit(pBright->getSRV(0, 1, slice, 1), pOutput->getRTV(0, slice, 1));
        }
        return;
    }

    auto vars = mpBlur->getRootVar();


    auto& dict = renderData.getDictionary();
    auto guardBand = dict.getValue("guardBand", 0);
    if(pBright->getArraySize() > 1)
    {
        guardBand = guardBand / int(std::sqrt((double)pBright->getArraySize()) + 0.5);
    }
    uint2 renderRes = { pBright->getWidth(), pBright->getHeight() };
    setGuardBandScissors(*mpBlur->getState(), renderRes, guardBand);
    // set scissor cb (is shared between both shaders)
    vars["ScissorCB"]["uvMin"] = dict.getValue("guardBand.uvMin", float2(0.0f));
    vars["ScissorCB"]["uvMax"] = dict.getValue("guardBand.uvMax", float2(1.0f));

    for(uint slice = 0; slice < pBright->getArraySize(); ++slice)
    {
        vars["gDepthTex"].setSrv(pDepth->getSRV(0, 1, slice, 1));
        //vars["gImportanceTex"].setSrv(pImportance->getSRV(0, 1, slice, 1));
        vars["gBrightTex"].setSrv(pBright->getSRV(0, 1, slice, 1));
        vars["gDarkTex"].setSrv(pDark->getSRV(0, 1, slice, 1));

        // blur in x
        vars["gSrcTex"].setSrv(pBright->getSRV(0, 1, slice, 1));
        mpFbo->attachColorTarget(pPingPong, 0, 0, slice, 1);
        vars["Direction"]["dir"] = float2(1.0f, 0.0f);
        mpBlur->execute(pRenderContext, mpFbo, false);

        // blur in y
        vars["gSrcTex"].setSrv(pPingPong->getSRV(0, 1, slice, 1));
        mpFbo->attachColorTarget(pOutput, 0, 0, slice, 1);
        vars["Direction"]["dir"] = float2(0.0f, 1.0f);
        mpBlur->execute(pRenderContext, mpFbo, false);
    }
}

void AOGuidedBlur::renderUI(Gui::Widgets& widget)
{
    widget.checkbox("Enabled", mEnabled);
    if (!mEnabled) return;

    if (widget.var("Kernel Radius", mKernelRadius, uint32_t(1), uint32_t(20))) requestRecompile();
}
