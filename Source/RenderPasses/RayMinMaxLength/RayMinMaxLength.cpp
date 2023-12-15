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
#include "RayMinMaxLength.h"

namespace
{
    const std::string kRayMin = "kRayMin";
    const std::string kRayMax = "kRayMax";
    const std::string kOut = "len";
    const std::string kShaderFilename = "RenderPasses/RayMinMaxLength/RayMinMaxLength.ps.slang";
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, RayMinMaxLength>();
}

RayMinMaxLength::RayMinMaxLength(ref<Device> pDevice, const Properties& props)
    : RenderPass(pDevice)
{
    mpPass = FullScreenPass::create(mpDevice, kShaderFilename);
    mpFbo = Fbo::create(mpDevice);
}

Properties RayMinMaxLength::getProperties() const
{
    return {};
}

RenderPassReflection RayMinMaxLength::reflect(const CompileData& compileData)
{
    mReady = false;

    // Define the required resources here
    RenderPassReflection reflector;
    reflector.addInput(kRayMin, "Minimum ray length");
    reflector.addInput(kRayMax, "Maximum ray length");

    reflector.addOutput(kOut, "Ray Length").format(ResourceFormat::R32Float);

    auto edge = compileData.connectedResources.getField(kRayMin);
    if(edge)
    {
        const auto srcWidth = edge->getWidth();
        const auto srcHeight = edge->getHeight();
        reflector.addOutput(kOut, "Ray Length").format(ResourceFormat::R32Float).texture2D(srcWidth, srcHeight);
        mReady = true;
    }

    return reflector;
}

void RayMinMaxLength::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    if (!mReady) throw std::runtime_error("RayMinMaxLength::compile - missing incoming reflection information");
}

void RayMinMaxLength::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto pRayMin = renderData[kRayMin]->asTexture();
    auto pRayMax = renderData[kRayMax]->asTexture();
    auto pOut = renderData[kOut]->asTexture();

    mpFbo->attachColorTarget(pOut, 0);
    auto var = mpPass->getRootVar();
    var["gRayMin"] = pRayMin;
    var["gRayMax"] = pRayMax;

    mpPass->execute(pRenderContext, mpFbo);
}

void RayMinMaxLength::renderUI(Gui::Widgets& widget)
{
}
