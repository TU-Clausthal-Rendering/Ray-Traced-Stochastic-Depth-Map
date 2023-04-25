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
#include "EnvMapPass.h"

#include "RenderGraph/RenderPassHelpers.h"

namespace
{
    const std::string kDepth = "depth";
    const std::string kColor = "color";

    const std::string kShaderFilename = "RenderPasses/EnvMapPass/EnvMapPass.ps.slang";
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, EnvMapPass>();
}

EnvMapPass::EnvMapPass(std::shared_ptr<Device> pDevice) : RenderPass(std::move(pDevice))
{
    mpFbo = Fbo::create(mpDevice.get());
}


EnvMapPass::SharedPtr EnvMapPass::create(std::shared_ptr<Device> pDevice, const Dictionary& dict)
{
    SharedPtr pPass = SharedPtr(new EnvMapPass(std::move(pDevice)));
    return pPass;
}

Dictionary EnvMapPass::getScriptingDictionary()
{
    return Dictionary();
}

RenderPassReflection EnvMapPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addInput(kDepth, "Non-linear depth buffer (currently unused)").bindFlags(Resource::BindFlags::DepthStencil);
    reflector.addOutput(kColor, "Envmap background")
        .format(ResourceFormat::RGBA32Float)
        .texture2D(mInputSize.x, mInputSize.y);
    return reflector;
}

void EnvMapPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto pDepth = renderData[kDepth]->asTexture();
    auto pColor = renderData[kColor]->asTexture();

    if (!RenderPassHelpers::isSameSize(pDepth, pColor))
    {
        mInputSize = uint2(pDepth->getWidth(), pDepth->getHeight());
        requestRecompile();
    }

    if (!mpScene) return;
    assert(mpPass);

    mpFbo->attachColorTarget(pColor, 0);

    mpPass["gScene"] = mpScene->getParameterBlock();
    mpPass->execute(pRenderContext, mpFbo);
}

void EnvMapPass::renderUI(Gui::Widgets& widget)
{
    
}

void EnvMapPass::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    mpScene = pScene;
    mpPass.reset();

    if (mpScene)
    {
        Program::Desc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderFilename).psEntry("main");
        desc.addTypeConformances(mpScene->getTypeConformances());
        desc.setShaderModel("6_5");
        Program::DefineList defines;
        defines.add(mpScene->getSceneDefines());
        mpPass = FullScreenPass::create(mpDevice, desc, defines);
    }
}

