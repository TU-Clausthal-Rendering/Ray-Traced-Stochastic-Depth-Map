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
#include "ForwardLighting.h"

namespace
{
    const char kShaderFile[] = "RenderPasses/ForwardLighting/ForwardLighting.3d.slang";

    const std::string kDepth = "depth";
    const std::string kColor = "color";
    const std::string kVisBuffer = "visibilityBuffer";

    const std::string kAmbientIntensity = "ambientIntensity";
    const std::string kEnvMapIntensity = "envMapIntensity";
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, ForwardLighting>();
}

ForwardLighting::ForwardLighting(ref<Device> pDevice) : RenderPass(std::move(pDevice))
{
    mpState = GraphicsState::create(mpDevice);

    mpFbo = Fbo::create(mpDevice);

    DepthStencilState::Desc dsDesc;
    dsDesc.setDepthWriteMask(false).setDepthFunc(DepthStencilState::Func::LessEqual);
    mpDsNoDepthWrite = DepthStencilState::create(dsDesc);
    mpState->setDepthStencilState(mpDsNoDepthWrite);
}

ref<ForwardLighting> ForwardLighting::create(ref<Device> pDevice, const Properties& dict)
{
    auto pPass = make_ref<ForwardLighting>(std::move(pDevice));
    for (const auto& [key, value] : dict)
    {
        if (key == kEnvMapIntensity) pPass->mEnvMapIntensity = value;
        else if (key == kAmbientIntensity) pPass->mAmbientIntensity = value;
        else logWarning("Unknown field '{}' in a ForwardLightingPass dictionary.", key);
    }
    return pPass;
}

Properties ForwardLighting::getProperties() const
{
    Properties d;
    d[kEnvMapIntensity] = mEnvMapIntensity;
    d[kAmbientIntensity] = mAmbientIntensity;
    return d;
}

RenderPassReflection ForwardLighting::reflect(const CompileData& compileData)
{
    
    RenderPassReflection reflector;
    reflector.addInput(kDepth, "Non-linear z-buffer");
    reflector.addInput(kVisBuffer, "Visibility buffer used for shadowing. Range is [0,1] where 0 means the pixel is fully-shadowed and 1 means the pixel is not shadowed at all").flags(RenderPassReflection::Field::Flags::Optional);
    reflector.addInputOutput(kColor, "Color texture");

    return reflector;
}

void ForwardLighting::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpScene) return;

    auto pColor = renderData[kColor]->asTexture();
    auto pDepth = renderData[kDepth]->asTexture();
    auto pVisBuffer = renderData[kVisBuffer]->asTexture();

    mpFbo->attachColorTarget(pColor, 0);
    mpFbo->attachDepthStencilTarget(pDepth);
    mpState->setFbo(mpFbo);

    mpVars->getRootVar()["visibilityBuffer"] = pVisBuffer;
    if(mDirty)
    {
        mpVars->getRootVar()["ConstantCB"]["gAmbientIntensity"] = mAmbientIntensity;
        mpVars->getRootVar()["ConstantCB"]["gEnvMapIntensity"] = mEnvMapIntensity;
        mDirty = false;
    }

    mpScene->rasterize(pRenderContext, mpState.get(), mpVars.get());
}

void ForwardLighting::renderUI(Gui::Widgets& widget)
{
    if (widget.var("Ambient Intensity", mAmbientIntensity, 0.f, 100.f, 0.1f)) mDirty = true;
    if (widget.var("Env Map Intensity", mEnvMapIntensity, 0.f, 100.f, 0.1f)) mDirty = true;
}

void ForwardLighting::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mpVars = nullptr;

    if (mpScene)
    {
        // create program
        Program::Desc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderFile).vsEntry("vsMain").psEntry("psMain");
        desc.addTypeConformances(mpScene->getTypeConformances());
        desc.setShaderModel("6_2");
        auto pProgram = GraphicsProgram::create(mpDevice, desc, mpScene->getSceneDefines());

        mpVars = GraphicsVars::create(mpDevice, pProgram->getReflector());
        mDirty = true;
        mpState->setProgram(pProgram);
    }
}


