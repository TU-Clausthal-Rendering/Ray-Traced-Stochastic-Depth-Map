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
#include "VarianceMap.h"

namespace
{
    const std::string kDepth = "linearZ";
    const std::string kNormal = "normalW"; // will be transformed to normalV

    const std::string kDepthOut = "depthOut";
    const std::string kNormalOut = "normalOut";

    const std::string kMipLevels = "mipLevels";

    const std::string kDepthInitShader = "RenderPasses/VarianceMap/DepthInit.ps.slang";
    const std::string kNormalInitShader = "RenderPasses/VarianceMap/NormalInit.ps.slang";
    const std::string kMipShader = "RenderPasses/VarianceMap/Mip.ps.slang";
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, VarianceMap>();
}

VarianceMap::VarianceMap(ref<Device> pDevice, const Dictionary& dict)
    : RenderPass(pDevice)
{
    for (const auto& [key, value] : dict)
    {
        if (key == kMipLevels) mMipLevels = value;
        else logWarning("Unknown field `" + key + "` in a VarianceMap dictionary");
    }
    
    mpFbo = Fbo::create(pDevice);

    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
    auto linearSampler = Sampler::create(pDevice, samplerDesc);

    mpDepthInitPass = FullScreenPass::create(pDevice, kDepthInitShader);
    mpDepthInitPass->getRootVar()["S"] = linearSampler;

    mpNormalInitPass = FullScreenPass::create(pDevice, kNormalInitShader);
    mpNormalInitPass->getRootVar()["S"] = linearSampler;

    mpMipPass = FullScreenPass::create(pDevice, kMipShader);
    mpMipPass->getRootVar()["S"] = linearSampler;
}

Dictionary VarianceMap::getScriptingDictionary()
{
    Dictionary dict;
    dict[kMipLevels] = mMipLevels;
    return dict;
}

RenderPassReflection VarianceMap::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addInput(kDepth, "linear depth buffer").bindFlags(ResourceBindFlags::ShaderResource).flags(RenderPassReflection::Field::Flags::Optional);
    reflector.addInput(kNormal, "world space normal buffer").bindFlags(ResourceBindFlags::ShaderResource).flags(RenderPassReflection::Field::Flags::Optional);
    reflector.addOutput(kDepthOut, "depth variance map").bindFlags(ResourceBindFlags::AllColorViews)
        .format(ResourceFormat::RG32Float)
        .texture2D(0, 0, 1, mMipLevels);
    reflector.addOutput(kNormalOut, "view-normal variance map (n.x, n.x^2, n.y, n.y^2)").bindFlags(ResourceBindFlags::AllColorViews)
        .format(ResourceFormat::RGBA16Float)
        .texture2D(0, 0, 1, mMipLevels);
        
    return reflector;
}

void VarianceMap::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto pDepthIn = renderData[kDepth]->asTexture();
    auto pNormalIn = renderData[kNormal]->asTexture();

    auto pDepthOut = renderData[kDepthOut]->asTexture();
    auto pNormalOut = renderData[kNormalOut]->asTexture();

    if(pDepthIn)
    {
        // init
        mpFbo->attachColorTarget(pDepthOut, 0, 0);
        mpDepthInitPass->getRootVar()["depthTex"] = pDepthIn;

        mpDepthInitPass->execute(pRenderContext, mpFbo);

        generateMips(pRenderContext, pDepthOut);
    }

    if(pNormalIn)
    {
        // init
        mpFbo->attachColorTarget(pNormalOut, 0, 0);
        mpNormalInitPass->getRootVar()["normalTex"] = pNormalIn;
        float4x4 viewMat = float4x4::identity();
        if (mpScene) viewMat = mpScene->getCamera()->getViewMatrix();
        mpNormalInitPass->getRootVar()["PerFrameCB"]["viewMat"] = viewMat;

        mpNormalInitPass->execute(pRenderContext, mpFbo);

        generateMips(pRenderContext, pNormalOut);
    }
}

void VarianceMap::renderUI(Gui::Widgets& widget)
{
    if (widget.var("Mip Levels", mMipLevels, -1, 14))
        requestRecompile();
}

void VarianceMap::generateMips(RenderContext* pRenderContext, const ref<Texture>& pTexture) const
{
    auto vars = mpMipPass->getRootVar();

    for(uint i = 1; i < pTexture->getMipCount(); ++i)
    {
        // bind the current mipmap level as render target
        mpFbo->attachColorTarget(pTexture, 0, i);
        // bind a single mip slice of the previous level
        vars["tex"].setSrv(pTexture->getSRV(i - 1, 1));

        mpMipPass->execute(pRenderContext, mpFbo);
    }
}
