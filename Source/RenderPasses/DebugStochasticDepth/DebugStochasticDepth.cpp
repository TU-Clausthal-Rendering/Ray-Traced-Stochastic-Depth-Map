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
#include "DebugStochasticDepth.h"

namespace
{
    const std::string kOut = "out";
    const std::string kShaderFile = "RenderPasses/DebugStochasticDepth/DebugStochasticDepth.slang";
    
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, DebugStochasticDepth>();
}

DebugStochasticDepth::DebugStochasticDepth(ref<Device> pDevice, const Properties& props)
    : RenderPass(pDevice)
{
    Program::Desc desc;
    desc.addShaderLibrary(kShaderFile).vsEntry("vsMain").psEntry("psMain");
    desc.setShaderModel("6_2");
    mpProgram = GraphicsProgram::create(mpDevice, desc);

    mpVars = GraphicsVars::create(mpDevice, mpProgram->getReflector());
    mpFbo = Fbo::create(mpDevice);


    mpState = GraphicsState::create(mpDevice);
    mpState->setProgram(mpProgram);
    mpState->setFbo(mpFbo);

    // disable depth testing
    DepthStencilState::Desc dsDesc;
    dsDesc.setDepthEnabled(false);
    mpState->setDepthStencilState(DepthStencilState::create(dsDesc));

    mpState->setVao(Vao::create(Vao::Topology::PointList, nullptr, Vao::BufferVec()));
}

Properties DebugStochasticDepth::getProperties() const
{
    return {};
}

RenderPassReflection DebugStochasticDepth::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;
    reflector.addInputOutput(kOut, "Output texture");
    return reflector;
}

void DebugStochasticDepth::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto& dict = renderData.getDictionary();

    if (!dict.keyExists("SD_MAP") || !mpScene) return;

    auto pSDMap = dict.getValue<ref<Texture>>("SD_MAP");
    if(!pSDMap) return;
    auto pSDCamera = dict.getValue<CameraData>("SD_CAMERA");
    auto jitter = dict.getValue("JITTER", true);
    auto sdGuard = dict.getValue("SD_GUARD", 0);

    auto pOut = renderData[kOut]->asTexture();
    //pRenderContext->clearTexture(pOut.get());
    mpFbo->attachColorTarget(pOut, 0);
    mpState->setFbo(mpFbo, true);
    uint2 res = uint2(pSDMap->getWidth(), pSDMap->getHeight());
    uint k = 0;
    switch(pSDMap->getFormat())
    {
    case ResourceFormat::R32Float:
    case ResourceFormat::R16Float:
        k = 1;
        break;
    case ResourceFormat::RG32Float:
    case ResourceFormat::RG16Float:
        k = 2;
        break;
    case ResourceFormat::RGBA32Float:
    case ResourceFormat::RGBA16Float:
        k = 4;
        break;
    }
    k = k * pSDMap->getArraySize();

    auto pCurCamera = mpScene->getCamera();

    auto vars = mpVars->getRootVar();
    vars["PerFrameCB"]["resolution"] = res;
    vars["PerFrameCB"]["numSamples"] = k;
    vars["gsDepthTex"] = pSDMap;
    vars["PerFrameCB"]["gCamera"].setBlob(pSDCamera);
    vars["PerFrameCB"]["sdGuard"] = sdGuard;
    vars["PerFrameCB"]["sdJitter"] = jitter;
    vars["PerFrameCB"]["newProjection"] = pCurCamera->getProjMatrix();
    auto prevToCurView = math::mul(pCurCamera->getViewMatrix(), math::inverse(pSDCamera.viewMat));
    vars["PerFrameCB"]["prevViewToCurView"] = prevToCurView;

    pRenderContext->draw(mpState.get(), mpVars.get(), res.x * res.y * k, 0);
}

void DebugStochasticDepth::renderUI(Gui::Widgets& widget)
{
}

void DebugStochasticDepth::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
}
