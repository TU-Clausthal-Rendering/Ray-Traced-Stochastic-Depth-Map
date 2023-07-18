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

    const std::string kIn = "ao2";
    //const std::string kImportance = "importance";
    const std::string kDepth = "lineardepth";
    const std::string kPingPong = "pingpong";

    const std::string kOutput = "color";

    const std::string kKernelRadius = "kernelRadius";
    const std::string kClampResults = "clampResults";
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, AOGuidedBlur>();
}

ref<AOGuidedBlur> AOGuidedBlur::create(ref<Device> pDevice, const Properties& dict)
{
    auto pPass = make_ref<AOGuidedBlur>(pDevice, dict);
    for (const auto& [key, value] : dict)
    {
        if (key == kKernelRadius) pPass->mKernelRadius = value;
        else if (key == kClampResults) pPass->mClampResults = value;
        else logWarning("Unknown field '" + key + "' in a AOGuidedBlur dictionary");
    }
    return pPass;
}

AOGuidedBlur::AOGuidedBlur(ref<Device> pDevice, const Properties& dict)
    : RenderPass(pDevice)
{
    mpFbo = Fbo::create(mpDevice);
    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(Sampler::Filter::Point, Sampler::Filter::Point, Sampler::Filter::Point).setAddressingMode(Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp);
    mpSampler = Sampler::create(mpDevice, samplerDesc);
}

Properties AOGuidedBlur::getProperties() const
{
    Properties dict;
    dict[kKernelRadius] = mKernelRadius;
    dict[kClampResults] = mClampResults;
    return dict;
}

RenderPassReflection AOGuidedBlur::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    mReady = false;
    reflector.addInput(kIn, "ao (bright, dark)").bindFlags(ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget).texture2D(0, 0, 1, 1, 0);
    reflector.addInput(kDepth, "linear depth").bindFlags(ResourceBindFlags::ShaderResource).texture2D(0,0,1,1,0);
    reflector.addInternal(kPingPong, "temporal result after first blur").bindFlags(ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget).format(ResourceFormat::RGBA8Unorm);
    reflector.addOutput(kOutput, "blurred ao").bindFlags(ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget).format(ResourceFormat::R8Unorm);

    // set correct input format and dimensions of the ping pong buffer
    auto edge = compileData.connectedResources.getField(kIn);
    if (edge)
    {
        const auto inputFormat = edge->getFormat();
        const auto srcWidth = edge->getWidth();
        const auto srcHeight = edge->getHeight();
        const auto srcArraySize = edge->getArraySize();

        mLastFormat = inputFormat;
        reflector.addInternal(kPingPong, "").texture2D(srcWidth, srcHeight, 1, 1, srcArraySize);
        reflector.addOutput(kOutput, "").texture2D(srcWidth, srcHeight, 1, 1, srcArraySize);

        mReady = true;
    }

    return reflector;
}

void AOGuidedBlur::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    if (!mReady) throw std::runtime_error("AOGuidedBlur::compile - missing incoming reflection information");

    DefineList defines = getShaderDefines();
    mpBlur = FullScreenPass::create(mpDevice, kShaderPath, defines);

    mpBlur->getRootVar()["gSampler"] = mpSampler;
}

void AOGuidedBlur::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto pIn = renderData[kIn]->asTexture();
    auto pPingPong = renderData[kPingPong]->asTexture();
    auto pDepth = renderData[kDepth]->asTexture();
    //auto pImportance = renderData[kImportance]->asTexture();

    auto pOutput = renderData[kOutput]->asTexture();

    if (!mEnabled)
    {
        // blit input to output
        for (uint slice = 0; slice < pIn->getArraySize(); ++slice)
        {
            pRenderContext->blit(pIn->getSRV(0, 1, slice, 1), pOutput->getRTV(0, slice, 1));
        }
        return;
    }

    auto vars = mpBlur->getRootVar();


    auto& dict = renderData.getDictionary();
    auto guardBand = dict.getValue("guardBand", 0);
    if(pIn->getArraySize() > 1)
    {
        guardBand = guardBand / int(std::sqrt((double)pIn->getArraySize()) + 0.5);
    }
    uint2 renderRes = { pIn->getWidth(), pIn->getHeight() };
    setGuardBandScissors(*mpBlur->getState(), renderRes, guardBand);
    // set scissor cb (is shared between both shaders)
    vars["ScissorCB"]["uvMin"] = dict.getValue("guardBand.uvMin", float2(0.0f));
    vars["ScissorCB"]["uvMax"] = dict.getValue("guardBand.uvMax", float2(1.0f));

    for(uint slice = 0; slice < pIn->getArraySize(); ++slice)
    {
        vars["gDepthTex"].setSrv(pDepth->getSRV(0, 1, slice, 1));
        //vars["gImportanceTex"].setSrv(pImportance->getSRV(0, 1, slice, 1));
        vars["gBrightDarkTex"].setSrv(pIn->getSRV(0, 1, slice, 1));

        // blur in x
        vars["gSrcTex"].setSrv(pIn->getSRV(0, 1, slice, 1));
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

    if (widget.var("Kernel Radius", mKernelRadius, uint32_t(1), uint32_t(20))) updateShaderDefines();
    //if (widget.checkbox("Clamp Results", mClampResults)) updateShaderDefines(); // currently happens automatically
    if (widget.checkbox("Local Deviation", mUseLocalDeviation)) updateShaderDefines();

    if (widget.dropdown("Output", mOutput)) updateShaderDefines();
}

void AOGuidedBlur::updateShaderDefines()
{
    if (!mpBlur) return;

    mpBlur->getProgram()->setDefines(getShaderDefines());
}

DefineList AOGuidedBlur::getShaderDefines() const
{
    DefineList defines;
    defines.add("KERNEL_RADIUS", std::to_string(mKernelRadius));
    defines.add("LOCAL_DEVIATION", mUseLocalDeviation ? "1" : "0");
    defines.add("CLAMP_RESULTS", std::to_string(mClampResults));
    defines.add("OUTPUT", std::to_string((uint)mOutput));
    return defines;
}
