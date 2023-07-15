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
#include "HBAO.h"
#include "../Utils/GuardBand/guardband.h"
#include <random>

namespace
{
    const char kDesc[] = "HBAO Plus von NVIDIA with interleaved texture access";

    const std::string kDepth = "depth";
    const std::string kDepth2 = "depth2";

    const std::string ksDepth = "stochasticDepth";
    const std::string kNormal = "normals";

    const std::string kAmbientMap = "ambientMap";

    const std::string kProgram = "RenderPasses/HBAO/HBAO.ps.slang";

    const Gui::DropdownList kDepthModeDropdown =
    {
        { (uint32_t)DepthMode::SingleDepth, "SingleDepth" },
        { (uint32_t)DepthMode::DualDepth, "DualDepth" },
        { (uint32_t)DepthMode::StochasticDepth, "StochasticDepth" },
    };

    const std::string kRadius = "radius";
    const std::string kDepthMode = "depthMode";
    const std::string kDepthBias = "depthBias";
    const std::string kExponent = "exponent";
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, HBAO>();
}

ref<HBAO> HBAO::create(ref<Device> pDevice, const Properties& props)
{
    auto pass = make_ref<HBAO>(pDevice, props);
    for (const auto& [key, value] : props)
    {
        if (key == kRadius) pass->mData.radius = value;
        else if (key == kDepthMode) pass->mDepthMode = value;
        else if (key == kDepthBias) pass->mData.NdotVBias = value;
        else if (key == kExponent) pass->mData.powerExponent = value;
        else logWarning("Unknown field '" + key + "' in a HBAOPlus dictionary");
    }
    return pass;
}

HBAO::HBAO(ref<Device> pDevice, const Properties& props)
    : RenderPass(pDevice)
{
    mpFbo = Fbo::create(mpDevice);
    mpPass = FullScreenPass::create(mpDevice, kProgram);
    // create sampler
    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(Sampler::Filter::Point, Sampler::Filter::Point, Sampler::Filter::Point).setAddressingMode(Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp);
    mpTextureSampler = Sampler::create(mpDevice, samplerDesc);

    setDepthMode(mDepthMode);
    setRadius(mData.radius);

    mNoiseTexture = genNoiseTexture();
}

Properties HBAO::getProperties() const
{
    Properties d;
    d[kRadius] = mData.radius;
    d[kDepthMode] = mDepthMode;
    d[kDepthBias] = mData.NdotVBias;
    d[kExponent] = mData.powerExponent;
    return d;
}

RenderPassReflection HBAO::reflect(const CompileData& compileData)
{
    // set correct size of output resource
    auto srcWidth = compileData.defaultTexDims.x;
    auto srcHeight = compileData.defaultTexDims.y;

    auto dstWidth = (srcWidth + 4 - 1) / 4;
    auto dstHeight = (srcHeight + 4 - 1) / 4;

    // Define the required resources here
    RenderPassReflection reflector;
    reflector.addInput(kDepth, "linear-depth (deinterleaved version)").bindFlags(ResourceBindFlags::ShaderResource).texture2D(0,0,1,1,16);
    reflector.addInput(kDepth2, "linear-depth2 (deinterleaved version)").bindFlags(ResourceBindFlags::ShaderResource).texture2D(0, 0, 1, 1, 16);

    reflector.addInput(kNormal, "normals").bindFlags(ResourceBindFlags::ShaderResource);//.texture2D(0, 0, 1, 1, 16);
    //reflector.addInput(ksDepth, "linearized stochastic depths").bindFlags(ResourceBindFlags::ShaderResource).texture2D(0, 0, 0, 1, 0);

    reflector.addOutput(kAmbientMap, "ambient occlusion (deinterleaved)").bindFlags(ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource)
        .texture2D(dstWidth, dstHeight, 1, 1, 16).format(ResourceFormat::RG8Unorm);

    return reflector;
}

void HBAO::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    mDirty = true;

    // static defines
    //auto sdepths = compileData.connectedResources.getField(ksDepth);
    //if (!sdepths) throw std::runtime_error("HBAOPlus::compile - missing incoming reflection information");

    //mpPass->getProgram()->addDefine("MSAA_SAMPLES", std::to_string(sdepths->getSampleCount()));
    //if (sdepths->getArraySize() == 1) mpPass->getProgram()->removeDefine("STOCHASTIC_ARRAY");
    //else mpPass->getProgram()->addDefine("STOCHASTIC_ARRAY");
}

void HBAO::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpScene) return;

    auto pDepthIn = renderData[kDepth]->asTexture();
    auto pDepth2In = renderData[kDepth2]->asTexture();
    //auto psDepth = renderData[ksDepth]->asTexture();
    auto pNormal = renderData[kNormal]->asTexture();
    auto pAmbientOut = renderData[kAmbientMap]->asTexture();

    if (!mEnabled)
    {
        // clear and return
        pRenderContext->clearTexture(pAmbientOut.get(), float4(1.0f));
        return;
    }

    auto vars = mpPass->getRootVar();

    if (mDirty)
    {
        // static data
        mData.resolution = float2(renderData.getDefaultTextureDims().x, renderData.getDefaultTextureDims().y);
        mData.invResolution = float2(1.0f) / mData.resolution;
        mData.noiseScale = mData.resolution / 4.0f; // noise texture is 4x4 resolution
        mData.invQuarterResolution = float2(1.0f) / float2(pDepthIn->getWidth(), pDepthIn->getHeight());
        vars["StaticCB"].setBlob(mData);


        vars["gTextureSampler"] = mpTextureSampler;
        mDirty = false;
    }

    auto pCamera = mpScene->getCamera().get();

    pCamera->setShaderData(vars["PerFrameCB"]["gCamera"]);
    vars["gNormalTex"] = pNormal;
    //vars["gsDepthTex"] = psDepth;

    auto& dict = renderData.getDictionary();
    auto guardBand = dict.getValue("guardBand", 0);
    setGuardBandScissors(*mpPass->getState(), uint2(pDepthIn->getWidth(), pDepthIn->getHeight()), guardBand / 4);

    for (int sliceIndex = 0; sliceIndex < 16; ++sliceIndex)
    {
        mpFbo->attachColorTarget(pAmbientOut, 0, 0, sliceIndex, 1);
        vars["gDepthTexQuarter"].setSrv(pDepthIn->getSRV(0, 1, sliceIndex, 1));
        vars["gDepthTex2Quarter"].setSrv(pDepth2In->getSRV(0, 1, sliceIndex, 1));
        vars["PerFrameCB"]["Rand"] = mNoiseTexture[sliceIndex];
        vars["PerFrameCB"]["quarterOffset"] = uint2(sliceIndex % 4, sliceIndex / 4);
        vars["PerFrameCB"]["sliceIndex"] = sliceIndex;

        mpPass->execute(pRenderContext, mpFbo, false);
    }
}

void HBAO::renderUI(Gui::Widgets& widget)
{
    widget.checkbox("Enabled", mEnabled);
    if (!mEnabled) return;

    float radius = mData.radius;
    if (widget.var("Radius", radius, 0.01f, FLT_MAX, 0.01f))
        setRadius(radius);

    if (widget.slider("Depth Bias", mData.NdotVBias, 0.0f, 0.5f)) mDirty = true;
    if (widget.slider("Power Exponent", mData.powerExponent, 1.0f, 4.0f)) mDirty = true;
    uint32_t depthMode = uint32_t(mDepthMode);
    if (widget.dropdown("Depth Mode", kDepthModeDropdown, depthMode))
        setDepthMode(DepthMode(depthMode));
}

void HBAO::setRadius(float r)
{
    mData.radius = r;
    mData.negInvRsq = -1.0f / (r * r);
    mDirty = true;
}

void HBAO::setDepthMode(DepthMode m)
{
    mDepthMode = m;
    mpPass->getProgram()->addDefine("DEPTH_MODE", std::to_string(uint32_t(m)));
}

std::vector<float4> HBAO::genNoiseTexture()
{
    std::vector<float4> data;
    data.resize(4u * 4u);

    // https://en.wikipedia.org/wiki/Ordered_dithering
    //const float ditherValues[] = { 0.0f, 8.0f, 2.0f, 10.0f, 12.0f, 4.0f, 14.0f, 6.0f, 3.0f, 11.0f, 1.0f, 9.0f, 15.0f, 7.0f, 13.0f, 5.0f };

    auto linearRand = [](float min, float max)
    {
        static std::mt19937 generator(0);
        std::uniform_real_distribution<float> distribution(min, max);
        return distribution(generator);
    };

    for (uint32_t i = 0; i < data.size(); i++)
    {
        // Random directions on the XY plane
        auto theta = linearRand(0.0f, 2.0f * 3.141f);
        //auto theta = ditherValues[i] / 16.0f * 2.0f * glm::pi<float>();
        auto r1 = linearRand(0.0f, 1.0f);
        auto r2 = linearRand(0.0f, 1.0f);
        data[i] = float4(sin(theta), cos(theta), r1, r2);
    }

    return data;
}
