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
#include "AOFlickerMask.h"

namespace
{
    const std::string kDepth = "linearZ";
    const std::string kNormals = "normalW";
    const std::string kMask = "mask";

    const std::string kShaderFile = "RenderPasses/AOFlickerMask/AOFlickerMask.ps.slang";
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, AOFlickerMask>();
}

AOFlickerMask::AOFlickerMask(ref<Device> pDevice, const Dictionary& dict)
    : RenderPass(pDevice)
{
    mpPass = FullScreenPass::create(mpDevice, kShaderFile);
    mpFbo = Fbo::create(mpDevice);
}

Dictionary AOFlickerMask::getScriptingDictionary()
{
    return Dictionary();
}

RenderPassReflection AOFlickerMask::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addInput(kDepth, "linear depths").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addInput(kNormals, "world space normals").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addOutput(kMask, "mask with stable pixels (1) and unstable/flickering (0)").format(ResourceFormat::R8Uint).bindFlags(ResourceBindFlags::RenderTarget);
    return reflector;
}

void AOFlickerMask::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpScene) return;
    
    auto pDepth = renderData[kDepth]->asTexture();
    auto pNormals = renderData[kNormals]->asTexture();
    auto pMask = renderData[kMask]->asTexture();

    mpFbo->attachColorTarget(pMask, 0);
    mpPass->getRootVar()["gLinearDepth"] = pDepth;
    mpPass->getRootVar()["gNormals"] = pNormals;
    mpScene->getCamera()->setShaderData(mpPass->getRootVar()["PerFrameCB"]["gCamera"]);
    mpPass->execute(pRenderContext, mpFbo);
}

void AOFlickerMask::renderUI(Gui::Widgets& widget)
{
}

void AOFlickerMask::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
 
}
