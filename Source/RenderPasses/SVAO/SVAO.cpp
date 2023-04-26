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
#include "SVAO.h"
#include "RenderGraph/RenderGraph.h"
#include "../Utils/GuardBand/guardband.h"

namespace
{
    const std::string kAmbientMap = "ao";
    const std::string kAoStencil = "stencil";
    //const std::string kAoStencil2 = "stencil2";
    const std::string kAccessStencil = "accessStencil";
    const std::string kGbufferDepth = "gbufferDepth";
    const std::string kDepth = "depth";
    const std::string kDepth2 = "depth2";
    const std::string kNormals = "normals";
    const std::string kMatDoubleSided = "doubleSided";
    const std::string kInternalStencil = "internalStencil";

    const std::string kRasterShader = "RenderPasses/SVAO/Raster.ps.slang";
    const std::string kRasterShader2 = "RenderPasses/SVAO/Raster2.ps.slang";
    const std::string kRayShader = "RenderPasses/SVAO/Ray.rt.slang";
    const std::string kStencilShader = "RenderPasses/SVAO/CopyStencil.ps.slang";

    const uint32_t kMaxPayloadSizePreventDarkHalos = 4 * 4;
    const uint32_t kMaxPayloadSizeDarkHalos = 4 * 3;

    // settings
    const std::string kRadius = "radius";
    const std::string kGuardBand = "guardBand";
    const std::string kPrimaryDepthMode = "primaryDepthMode";
    const std::string kSecondaryDepthMode = "secondaryDepthMode";
    const std::string kExponent = "exponent";
    const std::string kUseRayPipeline = "rayPipeline";
    const std::string kThickness = "thickness";
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, SVAO>();
}

SVAO::SVAO(std::shared_ptr<Device> pDevice) : RenderPass(std::move(pDevice))
{
    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(Sampler::Filter::Point, Sampler::Filter::Point, Sampler::Filter::Point).setAddressingMode(Sampler::AddressMode::Wrap, Sampler::AddressMode::Wrap, Sampler::AddressMode::Wrap);
    mpNoiseSampler = Sampler::create(mpDevice.get(), samplerDesc);

    samplerDesc.setFilterMode(Sampler::Filter::Point, Sampler::Filter::Point, Sampler::Filter::Point).setAddressingMode(Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp);
    mpTextureSampler = Sampler::create(mpDevice.get(), samplerDesc);

    mpFbo = Fbo::create(mpDevice.get());
    mpFbo2 = Fbo::create(mpDevice.get());

    mpNoiseTexture = genNoiseTexture();

    // set stencil to succeed if not equal to zero
    DepthStencilState::Desc stencil;
    stencil.setDepthEnabled(false);
    stencil.setDepthWriteMask(false);
    stencil.setStencilEnabled(true);
    stencil.setStencilOp(DepthStencilState::Face::FrontAndBack, DepthStencilState::StencilOp::Keep, DepthStencilState::StencilOp::Keep, DepthStencilState::StencilOp::Keep);
    stencil.setStencilFunc(DepthStencilState::Face::FrontAndBack, DepthStencilState::Func::NotEqual);
    stencil.setStencilRef(0);
    stencil.setStencilReadMask(1);
    stencil.setStencilWriteMask(0);
    mpDepthStencilState = DepthStencilState::create(stencil);

    // VAO settings will be loaded by first pass
    mpStencilPass = FullScreenPass::create(mpDevice, kStencilShader);
    stencil.setStencilOp(DepthStencilState::Face::FrontAndBack, DepthStencilState::StencilOp::Keep, DepthStencilState::StencilOp::Keep, DepthStencilState::StencilOp::Replace);
    stencil.setStencilFunc(DepthStencilState::Face::FrontAndBack, DepthStencilState::Func::Always);
    stencil.setStencilRef(1);
    stencil.setStencilReadMask(0);
    stencil.setStencilWriteMask(1);
    mpStencilPass->getState()->setDepthStencilState(DepthStencilState::create(stencil));
    mpStencilFbo = Fbo::create(mpDevice.get());

    mNeuralNet.load("../../NeuralNetVAO/net_relu");
    mNeuralNet2.load("../../NeuralNetVAO/net_relu_reg");
}

SVAO::SharedPtr SVAO::create(std::shared_ptr<Device> pDevice, const Dictionary& dict)
{
    SharedPtr pPass = SharedPtr(new SVAO(std::move(pDevice)));
    for (const auto& [key, value] : dict)
    {
        if (key == kRadius) pPass->mData.radius = value;
        else if (key == kPrimaryDepthMode) pPass->mPrimaryDepthMode = value;
        else if (key == kSecondaryDepthMode) pPass->mSecondaryDepthMode = value;
        else if (key == kExponent) pPass->mData.exponent = value;
        else if (key == kUseRayPipeline) pPass->mUseRayPipeline = value;
        else if (key == kGuardBand) pPass->mGuardBand = value;
        else if (key == kThickness) pPass->mData.thickness = value;
        else logWarning("Unknown field '" + key + "' in a VAONonInterleaved dictionary");
    }
    return pPass;
}

Dictionary SVAO::getScriptingDictionary()
{
    Dictionary d;
    d[kRadius] = mData.radius;
    d[kPrimaryDepthMode] = mPrimaryDepthMode;
    d[kSecondaryDepthMode] = mSecondaryDepthMode;
    d[kExponent] = mData.exponent;
    d[kUseRayPipeline] = mUseRayPipeline;
    d[kGuardBand] = mGuardBand;
    d[kThickness] = mData.thickness;
    return d;
}

RenderPassReflection SVAO::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    //reflector.addInput(kAoStencil, "(Depth-) Stencil Buffer for the ao mask").format(ResourceFormat::D32FloatS8X24);
    reflector.addInput(kGbufferDepth, "Non-Linear Depth from the G-Buffer").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addInput(kDepth, "Linear Depth-buffer").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addInput(kDepth2, "Linear Depth-buffer of second layer").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addInput(kNormals, "World space normals, [0, 1] range").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addInput(kMatDoubleSided, "Material double sided flag").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addOutput(kAmbientMap, "Ambient Occlusion (primary)").bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::RenderTarget).format(ResourceFormat::R8Unorm);
    reflector.addOutput(kAoStencil, "Stencil Bitmask for primary / secondary ao").bindFlags(ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource).format(ResourceFormat::R8Uint);
    //reflector.addInternal(kAoStencil2, "ping pong for stencil mask").bindFlags(ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource).format(ResourceFormat::R8Uint);
    reflector.addOutput(kAccessStencil, "Stencil Bitmask for secondary depth map accesses").bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource).format(ResourceFormat::R8Uint);
    reflector.addInternal(kInternalStencil, "internal stencil mask").format(ResourceFormat::D24UnormS8);
    return reflector;
}

void SVAO::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    mData.resolution = float2(compileData.defaultTexDims.x, compileData.defaultTexDims.y);
    mData.invResolution = float2(1.0f) / mData.resolution;
    mData.noiseScale = mData.resolution / 4.0f; // noise texture is 4x4 resolution

    mpRasterPass.reset(); // recompile passes
    mpRasterPass2.reset();
    mpRayProgram.reset();

    // create stochastic depth graph
    Dictionary sdDict;
    sdDict["SampleCount"] = msaa_sample;
    sdDict["Alpha"] = 0.2f;
    sdDict["linearize"] = true;
    sdDict["depthFormat"] = ResourceFormat::D24UnormS8;
    sdDict["CullMode"] = RasterizerState::CullMode::Back;
    mpStochasticDepthGraph = RenderGraph::create(mpDevice, "Stochastic Depth");
    auto pStochasticDepthPass = RenderPass::create("StochasticDepthMap", mpDevice, sdDict);
    mpStochasticDepthGraph->addPass(pStochasticDepthPass, "StochasticDepthMap");
    mpStochasticDepthGraph->markOutput("StochasticDepthMap.stochasticDepth");
    mpStochasticDepthGraph->setScene(mpScene);
}

void SVAO::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpScene) return;

    auto pNonLinearDepth = renderData[kGbufferDepth]->asTexture();
    auto pDepth = renderData[kDepth]->asTexture();
    auto pNormal = renderData[kNormals]->asTexture();
    auto pAoDst = renderData[kAmbientMap]->asTexture();
    auto pDepth2 = renderData[kDepth2]->asTexture();
    auto pMatDoubleSided = renderData[kMatDoubleSided]->asTexture();

    auto pAoMask = renderData[kAoStencil]->asTexture();
    //auto pAoMask2 = renderData[kAoStencil2]->asTexture();
    auto pAccessStencil = renderData[kAccessStencil]->asTexture();
    auto pInternalStencil = renderData[kInternalStencil]->asTexture();

    if (!mEnabled)
    {
        pRenderContext->clearTexture(pAoDst.get(), float4(1.0f));
        return;
    }

    if (!mpRasterPass || !mpRasterPass2 || !mpRayProgram) // this needs to be deferred because it needs the scene defines to compile
    {
        mNeuralNet.writeDefinesToFile("../RenderPasses/VAONonInterleaved/NeuralNetDefines.slangh");
        mNeuralNet2.writeDefinesToFile("../RenderPasses/VAONonInterleaved/NeuralNetDefines2.slangh");

        Program::DefineList defines;
        defines.add("PRIMARY_DEPTH_MODE", std::to_string(uint32_t(mPrimaryDepthMode)));
        defines.add("SECONDARY_DEPTH_MODE", std::to_string(uint32_t(mSecondaryDepthMode)));
        defines.add("MSAA_SAMPLES", std::to_string(msaa_sample)); // TODO update this from gui
        defines.add("PREVENT_DARK_HALOS", mPreventDarkHalos ? "1" : "0");
        defines.add("TRACE_OUT_OF_SCREEN", mTraceOutOfScreen ? "1" : "0");
        defines.add("TRACE_DOUBLE_ON_DOUBLE", mTraceDoubleOnDouble ? "1" : "0");
        defines.add("RAY_FILTER", mEnableRayFilter ? "1" : "0");
        defines.add(mpScene->getSceneDefines());
        // raster pass 1
        // TODO fix shaders for type conformance
        mpRasterPass = FullScreenPass::create(mpDevice, kRasterShader, defines);
        mpRasterPass->getProgram()->setTypeConformances(mpScene->getTypeConformances());

        // raster pass 2
        mpRasterPass2 = FullScreenPass::create(mpDevice, kRasterShader2, defines);
        mpRasterPass2->getProgram()->setTypeConformances(mpScene->getTypeConformances());
        mpRasterPass2->getState()->setDepthStencilState(mpDepthStencilState);

        // ray pass
        RtProgram::Desc desc;
        desc.addShaderLibrary(kRayShader);
        desc.setMaxPayloadSize(mPreventDarkHalos ? kMaxPayloadSizePreventDarkHalos : kMaxPayloadSizeDarkHalos);
        desc.setMaxAttributeSize(mpScene->getRaytracingMaxAttributeSize());
        desc.setMaxTraceRecursionDepth(1);
        desc.addTypeConformances(mpScene->getTypeConformances());

        RtBindingTable::SharedPtr sbt = RtBindingTable::create(1, 1, mpScene->getGeometryCount());
        sbt->setRayGen(desc.addRayGen("rayGen"));
        sbt->setMiss(0, desc.addMiss("miss"));
        sbt->setHitGroup(0, mpScene->getGeometryIDs(GeometryType::TriangleMesh), desc.addHitGroup("closestHit", "anyHit"));
        // TODO add remaining primitives
        mpRayProgram = RtProgram::create(mpDevice, desc, defines);
        mRayVars = RtProgramVars::create(mpDevice, mpRayProgram, sbt);
        mDirty = true;
    }

    if (mDirty)
    {
        // update data raster
        mpRasterPass["StaticCB"].setBlob(mData);
        mpRasterPass["gNoiseSampler"] = mpNoiseSampler;
        mpRasterPass["gTextureSampler"] = mpTextureSampler;
        mpRasterPass["gNoiseTex"] = mpNoiseTexture;
        // update data raster 2
        mpRasterPass2["StaticCB"].setBlob(mData);
        mpRasterPass2["gNoiseSampler"] = mpNoiseSampler;
        mpRasterPass2["gTextureSampler"] = mpTextureSampler;
        mpRasterPass2["gNoiseTex"] = mpNoiseTexture;
        // update data ray
        mRayVars["StaticCB"].setBlob(mData);
        mRayVars["gNoiseSampler"] = mpNoiseSampler;
        mRayVars["gTextureSampler"] = mpTextureSampler;
        mRayVars["gNoiseTex"] = mpNoiseTexture;

        // also clear ao texture if guard band changed
        pRenderContext->clearTexture(pAoDst.get(), float4(0.0f));
        mDirty = false;
    }

    auto accessStencilUAV = pAccessStencil->getUAV(0);
    if (mSecondaryDepthMode == DepthMode::StochasticDepth)
        pRenderContext->clearUAV(accessStencilUAV.get(), uint4(0u));

    mpFbo->attachColorTarget(pAoDst, 0);
    mpFbo->attachColorTarget(pAoMask, 1);
    //mpFbo->attachColorTarget(mEnableRayFilter ? pAoMask2 : pAoMask, 1);

    auto pCamera = mpScene->getCamera().get();
    pCamera->setShaderData(mpRasterPass["PerFrameCB"]["gCamera"]);
    mpRasterPass["PerFrameCB"]["invViewMat"] = inverse(pCamera->getViewMatrix());
    if (mPrimaryDepthMode == DepthMode::PerfectClassify)
    {
        // set raytracing data
        auto var = mpRasterPass->getRootVar();
        mpScene->setRaytracingShaderData(pRenderContext, var);
    }
    mpRasterPass["gDepthTex"] = pDepth;
    mpRasterPass["gDepthTex2"] = pDepth2;
    mpRasterPass["gNormalTex"] = pNormal;
    mpRasterPass["gMatDoubleSided"] = pMatDoubleSided;
    //mpRasterPass["gDepthAccess"] = pAccessStencil;
    mpRasterPass["gDepthAccess"].setUav(accessStencilUAV);

    setGuardBandScissors(*mpRasterPass->getState(), renderData.getDefaultTextureDims(), mGuardBand);
    {
        FALCOR_PROFILE(pRenderContext, "AO 1");
        mpRasterPass->execute(pRenderContext, mpFbo, false);
    }

    //if(mEnableRayFilter)
    //{
    //    FALCOR_PROFILE("RayFilter");
    //    mpRayFilter->execute(pRenderContext, pAoMask2, pAoMask); // result should be in pAoMask finally
    //}

    Texture::SharedPtr pStochasticDepthMap;

    //  execute stochastic depth map
    if (mSecondaryDepthMode == DepthMode::StochasticDepth)
    {
        FALCOR_PROFILE(pRenderContext, "Stochastic Depth");
        mpStochasticDepthGraph->setInput("StochasticDepthMap.depthMap", pNonLinearDepth);
        mpStochasticDepthGraph->setInput("StochasticDepthMap.stencilMask", pAccessStencil);
        mpStochasticDepthGraph->execute(pRenderContext);
        pStochasticDepthMap = mpStochasticDepthGraph->getOutput("StochasticDepthMap.stochasticDepth")->asTexture();
    }

    if (mUseRayPipeline) // RAY PIPELINE
    {
        // set raytracing data
        //mpScene->setRaytracingShaderData(pRenderContext, mRayVars);

        // set camera data
        auto pCamera = mpScene->getCamera().get();
        pCamera->setShaderData(mRayVars["PerFrameCB"]["gCamera"]);
        mRayVars["PerFrameCB"]["invViewMat"] = inverse(pCamera->getViewMatrix());
        mRayVars["PerFrameCB"]["guardBand"] = mGuardBand;

        // set textures
        mRayVars["gDepthTex"] = pDepth;
        mRayVars["gDepthTex2"] = pDepth2;
        mRayVars["gNormalTex"] = pNormal;
        mRayVars["gsDepthTex"] = pStochasticDepthMap;
        mRayVars["aoMask"] = pAoMask;
        mRayVars["gMatDoubleSided"] = pMatDoubleSided;
        //mRayVars["aoPrev"] = pAoDst; // src view
        mRayVars["output"] = pAoDst; // uav view


        uint3 dims = uint3(pAoDst->getWidth() - 2 * mGuardBand, pAoDst->getHeight() - 2 * mGuardBand, 1);
        mpScene->raytrace(pRenderContext, mpRayProgram.get(), mRayVars, uint3{ pAoDst->getWidth(), pAoDst->getHeight(), 1 });
    }
    else // RASTER PIPELINE
    {
        // copy stencil
        {
            FALCOR_PROFILE(pRenderContext, "copy stencil");
            auto dsv = pInternalStencil->getDSV();
            // clear stencil
            pRenderContext->clearDsv(dsv.get(), 0.0f, 0, false, true);
            mpStencilFbo->attachDepthStencilTarget(pInternalStencil);
            mpStencilPass["aoMask"] = pAoMask;
            mpStencilPass->execute(pRenderContext, mpStencilFbo);
            //pRenderContext->copySubresource(pInternalStencil.get(), 1, pAoMask.get(), 0); // <= don't do this, this results in a slow stencil
        }

        mpFbo2->attachDepthStencilTarget(pInternalStencil);
        mpFbo2->attachColorTarget(pAoDst, 0);

        // set raytracing data
        auto var = mpRasterPass2->getRootVar();
        mpScene->setRaytracingShaderData(pRenderContext, var);

        // set camera data
        auto pCamera = mpScene->getCamera().get();
        pCamera->setShaderData(mpRasterPass2["PerFrameCB"]["gCamera"]);
        mpRasterPass2["PerFrameCB"]["invViewMat"] = inverse(pCamera->getViewMatrix());

        // set textures
        mpRasterPass2["gDepthTex"] = pDepth;
        mpRasterPass2["gDepthTex2"] = pDepth2;
        mpRasterPass2["gNormalTex"] = pNormal;
        mpRasterPass2["gsDepthTex"] = pStochasticDepthMap;
        mpRasterPass2["gMatDoubleSided"] = pMatDoubleSided;
        mpRasterPass2["aoMask"] = pAoMask;
        mpRasterPass2["aoPrev"] = pAoDst;

        setGuardBandScissors(*mpRasterPass2->getState(), renderData.getDefaultTextureDims(), mGuardBand);

        {
            FALCOR_PROFILE(pRenderContext, "AO 2 (raster)");
            mpRasterPass2->execute(pRenderContext, mpFbo2, false);
        }
    }
}

void SVAO::renderUI(Gui::Widgets& widget)
{
    const Gui::DropdownList kPrimaryDepthModeDropdown =
    {
        { (uint32_t)DepthMode::SingleDepth, "SingleDepth" },
        { (uint32_t)DepthMode::DualDepth, "DualDepth" },
        { (uint32_t)DepthMode::MachineClassify, "MachineClassify" },
        { (uint32_t)DepthMode::MachinePredict, "MachinePredict" },
        { (uint32_t)DepthMode::PerfectClassify, "PerfectClassify" },
    };

    const Gui::DropdownList kSecondaryDepthModeDropdown =
    {
        { (uint32_t)DepthMode::StochasticDepth, "StochasticDepth" },
        { (uint32_t)DepthMode::Raytraced, "Raytraced" },
    };

    auto reset = false;

    widget.checkbox("Enabled", mEnabled);
    if (!mEnabled) return;

    if (widget.checkbox("Prevent Dark Halos", mPreventDarkHalos))
        reset = true;

    if (widget.var("Guard Band", mGuardBand, 0, 256))
        mDirty = true;

    uint32_t primaryDepthMode = (uint32_t)mPrimaryDepthMode;
    if (widget.dropdown("Primary Depth Mode", kPrimaryDepthModeDropdown, primaryDepthMode)) {
        mPrimaryDepthMode = (DepthMode)primaryDepthMode;
        reset = true;
    }

    if (mPrimaryDepthMode == DepthMode::MachineClassify)
    {
        if (widget.var("Machine Classify Threshold", mClassifyProbability, 0.01f, 0.99f))
        {
            mData.classifyThreshold = -glm::log(1.0f / mClassifyProbability - 1.0f);
            std::cout << "classify threshold: " << mClassifyProbability << ": " << mData.classifyThreshold << std::endl;
            mDirty = true;
        }
    }

    uint32_t secondaryDepthMode = (uint32_t)mSecondaryDepthMode;
    if (widget.dropdown("Secondary Depth Mode", kSecondaryDepthModeDropdown, secondaryDepthMode)) {
        mSecondaryDepthMode = (DepthMode)secondaryDepthMode;
        reset = true;
    }

    if (widget.checkbox("Ray Pipeline", mUseRayPipeline)) reset = true;

    if (widget.var("Sample Radius", mData.radius, 0.01f, FLT_MAX, 0.01f)) mDirty = true;

    if (widget.var("Thickness", mData.thickness, 0.0f, 1.0f, 0.1f)) {
        mDirty = true;
        //mData.exponent = glm::mix(1.6f, 1.0f, mData.thickness);
    }

    if (widget.var("Power Exponent", mData.exponent, 1.0f, 4.0f, 0.1f)) mDirty = true;

    if (mNeuralNet.renderUI(widget))
        reset = true;

    if (mNeuralNet2.renderUI(widget))
        reset = true;


    widget.separator();
    if (widget.checkbox("Enable Ray Filter", mEnableRayFilter)) reset = true;
    //if(mEnableRayFilter) mpRayFilter->renderUI(widget);

    if (widget.checkbox("Trace Out of Screen", mTraceOutOfScreen)) reset = true;
    widget.tooltip("If a sample point is outside of the screen, a ray is traced. Otherwise the closest sample from the border is used.");
    if (widget.checkbox("Trace Foliage Hits on Foliage Material", mTraceDoubleOnDouble)) reset = true;
    widget.tooltip("If disabled then no rays will be traced for double sided materials that hit double sided samples");

    if (widget.var("Fade End (Screen Space Radius)", mData.ssRadiusFadeEnd, 0.0f, 100.0f, 1.0f)) mDirty = true;
    widget.tooltip("radius in pixels where the ray tracing result is completely faded and only rasterization remains");

    if (widget.var("Fade Size (Screen Space Radius)", mData.ssRadiusFadeSize, 0.01f, 100.0f, 1.0f)) mDirty = true;
    widget.tooltip("The fade will be for a screen space radius in range [FadeEnd, FadeEnd + FadeSize]");


    if (reset) requestRecompile();
}

void SVAO::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    mpScene = pScene;
    mpRasterPass.reset(); // new scene defines => recompile
    mpRasterPass2.reset();
    mpRayProgram.reset();
    if (mpStochasticDepthGraph)
        mpStochasticDepthGraph->setScene(pScene);
}

Texture::SharedPtr SVAO::genNoiseTexture()
{
    static const int NOISE_SIZE = 4;
    std::vector<uint8_t> data;
    data.resize(NOISE_SIZE * NOISE_SIZE);

    // https://en.wikipedia.org/wiki/Ordered_dithering
    const float ditherValues[] = { 0.0f, 8.0f, 2.0f, 10.0f, 12.0f, 4.0f, 14.0f, 6.0f, 3.0f, 11.0f, 1.0f, 9.0f, 15.0f, 7.0f, 13.0f, 5.0f };

    std::srand(2346); // always use the same seed for the noise texture (linear rand uses std rand)
    for (uint32_t i = 0; i < data.size(); i++)
    {
        data[i] = uint8_t(ditherValues[i] / 16.0f * 255.0f);
    }

    return Texture::create2D(mpDevice.get(), NOISE_SIZE, NOISE_SIZE, ResourceFormat::R8Unorm, 1, 1, data.data());
}
