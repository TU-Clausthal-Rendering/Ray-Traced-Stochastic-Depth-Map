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
#include "CompressNormals.h"

namespace
{
    const std::string kNormalIn = "normalW";
    const std::string kNormalOut = "normalOut";
    const std::string kShaderFilename = "RenderPasses/CompressNormals/CompressNormals.ps.slang";

    const std::string kViewSpace = "viewSpace";
    const std::string k16Bit = "use16Bit";
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, CompressNormals>();
}

CompressNormals::CompressNormals(ref<Device> pDevice, const Properties& props)
    : RenderPass(pDevice)
{
    for (const auto& [key, value] : props)
    {
        if (key == kViewSpace) mViewSpace = value;
        else if (key == k16Bit) m16Bit = value;
        else logWarning("Unknown field '" + key + "' in a CompressNormals pass dictionary");
    }

    DefineList defines;
    defines.add("VIEW_SPACE", mViewSpace ? "1" : "0");
    defines.add("USE_16_BIT", m16Bit ? "1" : "0");
    mpPass = FullScreenPass::create(mpDevice, kShaderFilename, defines);
    mpFbo = Fbo::create(mpDevice);
}

Properties CompressNormals::getProperties() const
{
    Properties props;
    props[kViewSpace] = mViewSpace;
    props[k16Bit] = m16Bit;
    return props;
}

RenderPassReflection CompressNormals::reflect(const CompileData& compileData)
{
    auto format = ResourceFormat::R32Uint;
    if(m16Bit) format = ResourceFormat::R16Uint;

    // Define the required resources here
    RenderPassReflection reflector;
    reflector.addInput(kNormalIn, "World Space Normals");
    reflector.addOutput(kNormalOut, "Compressed Normals (Octa mapping)").format(format);
    return reflector;
}

void CompressNormals::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto pSrc = renderData[kNormalIn]->asTexture();
    auto pDst = renderData[kNormalOut]->asTexture();


    mpFbo->attachColorTarget(pDst, 0);
    auto vars = mpPass->getRootVar();
    vars["gNormals"] = pSrc;

    if(mViewSpace && mpScene)
    {
        vars["PerFrameCB"]["gViewMat"] = mpScene->getCamera()->getViewMatrix();
    }

    mpPass->execute(pRenderContext, mpFbo);
}

void CompressNormals::renderUI(Gui::Widgets& widget)
{
    if(widget.checkbox("View Space", mViewSpace))
    {
        mpPass->getProgram()->addDefine("VIEW_SPACE", mViewSpace ? "1" : "0");
    }
    widget.tooltip("If checked, normals will be transformed to view space before compression. Otherwise they will stay in world space");
    if(widget.checkbox("16 bit", m16Bit))
    {
        mpPass->getProgram()->addDefine("USE_16_BIT", m16Bit ? "1" : "0");
        requestRecompile();
    }
}

void CompressNormals::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
}
