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
#include "ReconstructNormals.h"

namespace
{
    const char kShaderFile[] = "RenderPasses/ReconstructNormals/ReconstructNormals.ps.slang";
    const std::string kDepthIn = "linearZ";
    const std::string kNormalOut = "normal";

    const std::string kCompress = "compress";
    const std::string k16Bit = "use16Bit";
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, ReconstructNormals>();
}

ReconstructNormals::ReconstructNormals(ref<Device> pDevice, const Properties& props)
    : RenderPass(pDevice)
{
    for (const auto& [key, value] : props)
    {
        if (key == kCompress) mCompress = value;
        else if (key == k16Bit) m16Bit = value;
        else logWarning("Unknown field `" + key + "` in a ReconstructNormals pass dictionary");
    }

    DefineList defines;
    defines.add("COMPRESS", mCompress ? "1" : "0");
    defines.add("USE_16_BIT", m16Bit ? "1" : "0");
    mpPass = FullScreenPass::create(pDevice, kShaderFile, defines);
    mpFbo = Fbo::create(pDevice);

    // create point sampler
    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(Sampler::Filter::Point, Sampler::Filter::Point, Sampler::Filter::Point);
    mpPass->getRootVar()["gSampler"] = Sampler::create(pDevice, samplerDesc);
}

Properties ReconstructNormals::getProperties() const
{
    Properties props;
    props[kCompress] = mCompress;
    props[k16Bit] = m16Bit;
    return props;
}

RenderPassReflection ReconstructNormals::reflect(const CompileData& compileData)
{
    
    auto format = ResourceFormat::RGBA16Snorm;
    if(mCompress)
    {
        format = ResourceFormat::R32Uint;
        if (m16Bit) format = ResourceFormat::R16Uint;
    }

    RenderPassReflection reflector;
    reflector.addInput(kDepthIn, "Depth buffer").bindFlags(Resource::BindFlags::ShaderResource);
    reflector.addOutput(kNormalOut, "Reconstructed normal (octa compressed)").bindFlags(Resource::BindFlags::RenderTarget).format(format);
    return reflector;
}

void ReconstructNormals::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpScene) return;

    auto pDepth = renderData[kDepthIn]->asTexture();
    auto pNormal = renderData[kNormalOut]->asTexture();

    mpFbo->attachColorTarget(pNormal, 0);
    auto var = mpPass->getRootVar();
    var["gDepth"] = pDepth;

    mpScene->getCamera()->setShaderData(var["StaticCB"]["gCamera"]);
    var["StaticCB"]["duv"] = float2(1.0f) / float2(pDepth->getWidth(), pDepth->getHeight());

    mpPass->execute(pRenderContext, mpFbo);
}

void ReconstructNormals::renderUI(Gui::Widgets& widget)
{
    if(widget.checkbox("Compress", mCompress))
    {
        mpPass->getProgram()->addDefine("COMPRESS", mCompress ? "1" : "0");
        requestRecompile();
    }

    if (!mCompress) return;

    if (widget.checkbox("16 bit", m16Bit))
    {
        mpPass->getProgram()->addDefine("USE_16_BIT", m16Bit ? "1" : "0");
        requestRecompile();
    }
}
