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
#include "DeferredLighting.h"

namespace
{
    const std::string kPosition = "posW";
    const std::string kNormal = "normW";
    const std::string kDiffuse = "diffuseOpacity";
    const std::string kSpecular = "specularRough";
    const std::string kEmissive = "emissive";
    const std::string kColorOut = "color";
    const std::string kShaderFilename = "RenderPasses/DeferredLighting/DeferredLighting.ps.slang";
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, DeferredLighting>();
}

DeferredLighting::DeferredLighting(ref<Device> pDevice) : RenderPass(std::move(pDevice))
{
    mpFbo = Fbo::create(mpDevice);
}

ref<DeferredLighting> DeferredLighting::create(ref<Device> pDevice, const Properties& dict)
{
    auto pPass = make_ref<DeferredLighting>(std::move(pDevice));
    return pPass;
}

Properties DeferredLighting::getProperties() const
{
    return Properties();
}

RenderPassReflection DeferredLighting::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addInput(kPosition, "World-space position");
    reflector.addInput(kNormal, "World-space normal");
    reflector.addInput(kDiffuse, "Diffuse color and opacity");
    reflector.addInput(kSpecular, "Specular color and roughness");
    reflector.addInput(kEmissive, "Emissive color");
    reflector.addOutput(kColorOut, "Color buffer").format(ResourceFormat::RGBA32Float);
    return reflector;
}

void DeferredLighting::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpScene) return;
    assert(mpPass);

    auto pPos = renderData[kPosition]->asTexture();
    auto pNorm = renderData[kNormal]->asTexture();
    auto pDiffuse = renderData[kDiffuse]->asTexture();
    auto pSpecular = renderData[kSpecular]->asTexture();
    auto pEmissive = renderData[kEmissive]->asTexture();

    auto pDst = renderData[kColorOut]->asTexture();
    mpFbo->attachColorTarget(pDst, 0);

    auto vars = mpPass->getRootVar();
    vars["gScene"] = mpScene->getParameterBlock();

    vars["gPos"] = pPos;
    vars["gNorm"] = pNorm;
    vars["gDiff"] = pDiffuse;
    vars["gSpec"] = pSpecular;
    vars["gEmissive"] = pEmissive;

    mpPass->execute(pRenderContext, mpFbo);
}

void DeferredLighting::renderUI(Gui::Widgets& widget)
{
    
}

void DeferredLighting::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
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
        DefineList defines;
        defines.add(mpScene->getSceneDefines());
        mpPass = FullScreenPass::create(mpDevice, desc, defines);
    }
}
