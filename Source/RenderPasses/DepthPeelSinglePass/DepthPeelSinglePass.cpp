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
#include "DepthPeelSinglePass.h"
#include "Core/API/NativeHandleTraits.h"
#include "Core/API/NativeFormats.h"
#include <d3d12.h>
#include <wrl/client.h>

namespace
{
    const std::string kProgramFile = "RenderPasses/DepthPeelSinglePass/DepthPeelSinglePass.3d.slang";

    const std::string kDepthArray = "depthArray";
    const std::string kDepthPrimary = "depthPrimary";
    const std::string kDepthSecondary = "depthSecondary";
    const std::string kDepthFormat = "depthFormat";
    const std::string kUseAlphaTest = "useAlphaTest";
    const std::string kCullMode = "cullMode";

    const std::string kPrevDepth = "prevDepth";

    const Gui::DropdownList kCullModeList =
    {
        { (uint32_t)RasterizerState::CullMode::None, "None" },
        { (uint32_t)RasterizerState::CullMode::Back, "Back" },
        { (uint32_t)RasterizerState::CullMode::Front, "Front" },
    };
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, DepthPeelSinglePass>();
}

ref<DepthPeelSinglePass> DepthPeelSinglePass::create(ref<Device> pDevice, const Properties& props)
{
    auto pPass = make_ref<DepthPeelSinglePass>(pDevice, props);
    for (const auto& [key, value] : props)
    {
        if (key == kDepthFormat) pPass->mDepthFormat = value;
        else if (key == kUseAlphaTest) pPass->mUseAlphaTest = value;
        else if (key == kCullMode) pPass->mCullMode = value;
        else logWarning("Unknown field '" + key + "' in a DepthPass dictionary");
    }
    return pPass;
}

DepthPeelSinglePass::DepthPeelSinglePass(ref<Device> pDevice, const Properties& props)
    : RenderPass(pDevice)
{
    mpState = GraphicsState::create(mpDevice);
    mpFbo = Fbo::create(mpDevice);

    auto pRawDevice = mpDevice->getNativeHandle().as<ID3D12Device*>();
    D3D12_FEATURE_DATA_D3D12_OPTIONS d3d12Options = {};
    pRawDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &d3d12Options, sizeof(d3d12Options));

    if(!d3d12Options.VPAndRTArrayIndexFromAnyShaderFeedingRasterizerSupportedWithoutGSEmulation)
    {
        logWarning("DepthPeelSinglePass requires VPAndRTArrayIndexFromAnyShaderFeedingRasterizerSupportedWithoutGSEmulation");
    }
}

Properties DepthPeelSinglePass::getProperties() const
{
    Properties d;
    d[kDepthFormat] = mDepthFormat;
    d[kUseAlphaTest] = mUseAlphaTest;
    d[kCullMode] = mCullMode;
    return d;
}

RenderPassReflection DepthPeelSinglePass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addOutput(kDepthArray, "Depth-buffer").bindFlags(ResourceBindFlags::AllDepthViews).format(mDepthFormat).texture2D(mOutputSize.x, mOutputSize.y, 1, 1, 2);
    reflector.addOutput(kDepthPrimary, "1st Depth-buffer").bindFlags(Resource::BindFlags::AllDepthViews).format(mDepthFormat).texture2D(mOutputSize.x, mOutputSize.y, 1, 1, 1);
    reflector.addOutput(kDepthSecondary, "2nd Depth-buffer").bindFlags(Resource::BindFlags::AllDepthViews).format(mDepthFormat).texture2D(mOutputSize.x, mOutputSize.y, 1, 1, 1);
    reflector.addInternal(kPrevDepth, "prev primary depth").bindFlags(Resource::BindFlags::AllDepthViews).format(mDepthFormat).texture2D(mOutputSize.x, mOutputSize.y, 1, 1, 1);
    return reflector;
}

void DepthPeelSinglePass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto pDepthArr = renderData[kDepthArray]->asTexture();
    ref<Texture> pDepthPrimary;
    if(renderData[kDepthPrimary])
        pDepthPrimary = renderData[kDepthPrimary]->asTexture();
    ref<Texture> pDepthSecondary;
    if(renderData[kDepthSecondary])
        pDepthSecondary = renderData[kDepthSecondary]->asTexture();
    auto pPrevDepth = renderData[kPrevDepth]->asTexture();

    mpFbo->attachDepthStencilTarget(pDepthArr, 0, 0, 2);

    mpState->setFbo(mpFbo);
    pRenderContext->clearDsv(pDepthArr->getDSV(0, 0, 2).get(), 1, 0);

    if (mpScene)
    {
        auto vars = mpVars->getRootVar();
        vars["PerFrameCB"]["gFrameDim"] = uint2(pDepthArr->getWidth(), pDepthArr->getHeight());
        vars["prevDepthTex"] = pPrevDepth;

        //mpState->getProgram()->addDefine("USE_ALPHA_TEST", mUseAlphaTest ? "1" : "0");
        mpScene->rasterize(pRenderContext, mpState.get(), mpVars.get(), mCullMode);

        // copy first layer to kDepthPrimary
        if (pDepthPrimary)
            //pRenderContext->blit(pDepthArr->getSRV(0, 1, 0, 1), pDepthPrimary->getRTV());
            pRenderContext->copySubresource(pDepthPrimary.get(), 0, pDepthArr.get(), 0);
        if(pDepthSecondary)
            //pRenderContext->blit(pDepthArr->getSRV(0, 1, 1, 1), pDepthSecondary->getRTV());
            pRenderContext->copySubresource(pDepthSecondary.get(), 0, pDepthArr.get(), 1);

        // always copy prev depth
        pRenderContext->copySubresource(pPrevDepth.get(), 0, pDepthArr.get(), 0);
    }
}

void DepthPeelSinglePass::renderUI(Gui::Widgets& widget)
{
    static const Gui::DropdownList kDepthFormats =
    {
        { (uint32_t)ResourceFormat::D16Unorm, "D16Unorm"},
        { (uint32_t)ResourceFormat::D32Float, "D32Float" },
        { (uint32_t)ResourceFormat::D24UnormS8, "D24UnormS8" },
        { (uint32_t)ResourceFormat::D32FloatS8X24, "D32FloatS8X24" },
    };

    uint32_t depthFormat = (uint32_t)mDepthFormat;
    if (widget.dropdown("Buffer Format", kDepthFormats, depthFormat))
        setDepthBufferFormat(ResourceFormat(depthFormat));

    uint32_t cullMode = (uint32_t)mCullMode;
    if (widget.dropdown("Cull mode", kCullModeList, cullMode))
        mCullMode = (RasterizerState::CullMode)cullMode;
}

void DepthPeelSinglePass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mpVars = nullptr;

    if (mpScene)
    {
        Program::Desc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kProgramFile).vsEntry("vsMain").psEntry("psMain");
        desc.addTypeConformances(mpScene->getTypeConformances());
        desc.setShaderModel("6_2");
        auto pProgram = GraphicsProgram::create(mpDevice, desc, mpScene->getSceneDefines());


        pProgram->addDefine("USE_ALPHA_TEST", mUseAlphaTest ? "1" : "0");
        mpVars = GraphicsVars::create(mpDevice, pProgram->getReflector());
        mpState->setProgram(pProgram);
    }

}

DepthPeelSinglePass& DepthPeelSinglePass::setDepthBufferFormat(ResourceFormat format)
{
    if (isDepthStencilFormat(format) == false)
    {
        logWarning("DepthPeelSinglePass buffer format must be a depth-stencil format");
    }
    else
    {
        mDepthFormat = format;
        requestRecompile();
    }
    return *this;
}
