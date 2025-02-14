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
#include "RayShadow.h"

#include "Core/API/Sampler.h"
#include "RenderGraph/RenderPassHelpers.h"

namespace
{
    const std::string kPos = "posW";
    const std::string kNormalW = "normalW";
    const std::string kVisibility = "visibility";

    const std::string kRasterShader = "RenderPasses/RayShadow/RayShadow.ps.slang";
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, RayShadow>();
}

RayShadow::RayShadow(ref<Device> pDevice) : RenderPass(std::move(pDevice))
{
    mpFbo = Fbo::create(mpDevice);

    Sampler::Desc d;
    //d.setFilterMode(Sampler::Filter::Point, Sampler::Filter::Point, Sampler::Filter::Point);
    d.setAddressingMode(Sampler::AddressMode::Border, Sampler::AddressMode::Border, Sampler::AddressMode::Border);
    d.setBorderColor(float4(0.0f));
    //d.setMaxAnisotropy(16);
    
    mpSampler = Sampler::create(mpDevice, d);
}

ref<RayShadow> RayShadow::create(ref<Device> pDevice, const Properties& dict)
{
    auto pPass = make_ref<RayShadow>(std::move(pDevice));
    return pPass;
}

Properties RayShadow::getProperties() const
{
    return Properties();
}

RenderPassReflection RayShadow::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;

    
    reflector.addInput(kPos, "Pre-initialized scene depth buffer")
        .bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addInput(kNormalW, "Pre-initialized scene normal buffer").bindFlags(ResourceBindFlags::ShaderResource);

    reflector.addOutput(kVisibility, "Visibility map. Values are [0,1] where 0 means the pixel is completely shadowed and 1 means it's not shadowed at all")
        .format(ResourceFormat::R8Unorm)
        .texture2D(mInputSize.x, mInputSize.y, 1, 1, mLightCount)
        .bindFlags(ResourceBindFlags::RenderTarget);
    return reflector;
}

void RayShadow::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto pPos = renderData[kPos]->asTexture();
    auto pNormal = renderData[kNormalW]->asTexture();
    auto pVisibility = renderData[kVisibility]->asTexture();
    
    if (!RenderPassHelpers::isSameSize(pPos, pVisibility))
    {
        mInputSize = uint2(pPos->getWidth(), pPos->getHeight());
        requestRecompile();
    }

    // clear visibility texture
    pRenderContext->clearTexture(pVisibility.get(), float4(1, 1, 1, 1));
    if (!mpScene) return;

    if (!mpPass)
    {
        Program::Desc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kRasterShader).psEntry("main");
        desc.addTypeConformances(mpScene->getTypeConformances());
        desc.setShaderModel("6_5");
        DefineList defines;
        auto rayConeSpread = mpScene->getCamera()->computeScreenSpacePixelSpreadAngle(renderData.getDefaultTextureDims().y);
        defines.add("RAY_CONE_SPREAD", std::to_string(rayConeSpread));
        defines.add(mpScene->getSceneDefines());
        defines.add("USE_RAYCONES", mRayCones ? "1" : "0");
        defines.add("RAY_CONE_SHADOW", std::to_string(int(mRayConeShadow)));
        mpPass = FullScreenPass::create(mpDevice, desc, defines);
        auto vars = mpPass->getRootVar();
        vars["gSoftShadowSampler"] = mpSampler;
    }

    auto var = mpPass->getRootVar();
    var["gPos"] = pPos;
    var["gNormal"] = pNormal;
    var["PerLightBuffer"]["gPointLightClip"] = mPointLightClip;
    var["PerLightBuffer"]["gLodBias"] = mLodBias;
    var["PerLightBuffer"]["gDiminishBorder"] = mDiminishBorder;

    // raytracing data
    mpScene->setRaytracingShaderData(pRenderContext, var);

    auto nLights = std::min(mLightCount, (int)mpScene->getLightCount());
    for(int i = 0; i < nLights; i++)
    {
        var["PerLightBuffer"]["gLightIndex"] = i;
        mpFbo->attachColorTarget(pVisibility, 0, 0, i, 1);
        mpPass->execute(pRenderContext, mpFbo);
    }
}

void RayShadow::renderUI(Gui::Widgets& widget)
{
    if(widget.checkbox("Ray Cones", mRayCones))
    {
        mpPass.reset();
    }
    if(mRayCones)
    {
        if (widget.dropdown("Ray Cone Shadows", mRayConeShadow))
            mpPass.reset();
    }
    widget.var("LOD Bias", mLodBias, -16.0f, 16.0f, 0.5f);

    if(widget.var("Lights", mLightCount, 1, 128))
    {
        requestRecompile();
    }

    widget.var("Point Light Clip", mPointLightClip, 0.0f);

    widget.checkbox("Diminish Border", mDiminishBorder);
    widget.tooltip("Blends out texture fetches near the border to prevent showing the quadratic shape of the texture in higher mip levels");
}

void RayShadow::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mpPass.reset();

    if(mpScene)
    {
        mLightCount = std::min((int)mpScene->getLightCount(), 128);
    }
}
