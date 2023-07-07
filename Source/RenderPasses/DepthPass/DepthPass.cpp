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
#include "DepthPass.h"

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, DepthPass>();
}

namespace
{
    const std::string kProgramFile = "RenderPasses/DepthPass/DepthPass.ps.slang";

    const std::string kDepth = "depth";
    const std::string kDepthFormat = "depthFormat";
    const std::string kUseAlphaTest = "useAlphaTest";
    const std::string kCullMode = "cullMode";

    const Gui::DropdownList kCullModeList =
    {
        { (uint32_t)RasterizerState::CullMode::None, "None" },
        { (uint32_t)RasterizerState::CullMode::Back, "Back" },
        { (uint32_t)RasterizerState::CullMode::Front, "Front" },
    };
}

DepthPass::DepthPass(ref<Device> pDevice) : RenderPass(std::move(pDevice))
{
    mpState = GraphicsState::create(mpDevice);
    mpFbo = Fbo::create(mpDevice);
}

ref<DepthPass> DepthPass::create(ref<Device> pDevice, const Properties& dict)
{
    auto pPass = make_ref<DepthPass>(std::move(pDevice));
    for (const auto& [key, value] : dict)
    {
        if (key == kDepthFormat) pPass->setDepthBufferFormat(value);
        else if (key == kUseAlphaTest) pPass->mUseAlphaTest = value;
        else if (key == kCullMode) pPass->mCullMode = value;
        else logWarning("Unknown field '" + key + "' in a DepthPass dictionary");
    }
    return pPass;
}

DepthPass& DepthPass::setDepthBufferFormat(ResourceFormat format)
{
    if (isDepthStencilFormat(format) == false)
    {
        logWarning("DepthPass buffer format must be a depth-stencil format");
    }
    else
    {
        mDepthFormat = format;
        requestRecompile();
    }
    return *this;
}

Properties DepthPass::getProperties() const
{
    Properties d;
    d[kDepthFormat] = mDepthFormat;
    d[kUseAlphaTest] = mUseAlphaTest;
    d[kCullMode] = mCullMode;
    return d;
}

RenderPassReflection DepthPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addOutput(kDepth, "Depth-buffer").bindFlags(Resource::BindFlags::DepthStencil).format(mDepthFormat).texture2D(mOutputSize.x, mOutputSize.y);
    return reflector;
}

void DepthPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    const auto& pDepth = renderData[kDepth]->asTexture();
    mpFbo->attachDepthStencilTarget(pDepth);

    mpState->setFbo(mpFbo);
    pRenderContext->clearDsv(pDepth->getDSV().get(), 1, 0);

    if (mpScene)
    {
        //mpState->getProgram()->addDefine("USE_ALPHA_TEST", mUseAlphaTest ? "1" : "0");
        mpScene->rasterize(pRenderContext, mpState.get(), mpVars.get(), mCullMode);
    }
}

void DepthPass::renderUI(Gui::Widgets& widget)
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

void DepthPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
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


