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
    const std::string kGbufferDepth = "gbufferDepth";
    const std::string kDepth = "depth";
    const std::string kDepth2 = "depth2";
    const std::string kNormals = "normals";
    const std::string kColor = "color";

    const std::string kInternalStencil = "internalStencil";
    // ray bounds for the stochastic depth map RT
    const std::string kInternalRayMin = "internalRayMin";
    const std::string kInternalRayMax = "internalRayMax";

    const std::string kRasterShader = "RenderPasses/SVAO/SVAORaster.ps.slang";
    const std::string kRasterShader2 = "RenderPasses/SVAO/SVAORaster2.ps.slang";
    const std::string kRayShader = "RenderPasses/SVAO/Ray.rt.slang";
    const std::string kStencilShader = "RenderPasses/SVAO/CopyStencil.ps.slang";

    const uint32_t kMaxPayloadSizePreventDarkHalos = 4 * 4;

    // settings
    const std::string kRadius = "radius";
    const std::string kPrimaryDepthMode = "primaryDepthMode";
    const std::string kSecondaryDepthMode = "secondaryDepthMode";
    const std::string kExponent = "exponent";
    const std::string kUseRayPipeline = "rayPipeline";
    const std::string kThickness = "thickness";
    const std::string kImportanceThreshold = "importanceThreshold";
    const std::string kTargetTime = "targetTime";
    const std::string kStochMapDivisor = "stochMapDivisor"; // stochastic depth map resolution divisor
    const std::string kDualAo = "dualAO";
    const std::string kAlphaTest = "alphaTest";
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, SVAO>();
}

SVAO::SVAO(ref<Device> pDevice) : RenderPass(std::move(pDevice))
{
    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(Sampler::Filter::Point, Sampler::Filter::Point, Sampler::Filter::Point).setAddressingMode(Sampler::AddressMode::Wrap, Sampler::AddressMode::Wrap, Sampler::AddressMode::Wrap);
    mpNoiseSampler = Sampler::create(mpDevice, samplerDesc);

    samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Point).setAddressingMode(Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp);
    //samplerDesc.setFilterMode(Sampler::Filter::Point, Sampler::Filter::Point, Sampler::Filter::Point).setAddressingMode(Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp);
    mpTextureSampler = Sampler::create(mpDevice, samplerDesc);

    mpFbo = Fbo::create(mpDevice);
    mpFbo2 = Fbo::create(mpDevice);

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
    stencil.setStencilOp(DepthStencilState::Face::FrontAndBack, DepthStencilState::StencilOp::Keep, DepthStencilState::StencilOp::Keep, DepthStencilState::StencilOp::Increase);
    stencil.setStencilFunc(DepthStencilState::Face::FrontAndBack, DepthStencilState::Func::Always);
    stencil.setStencilRef(1); // does not work => using increase instead
    stencil.setStencilReadMask(1);
    stencil.setStencilWriteMask(1);
    mpStencilPass->getState()->setDepthStencilState(DepthStencilState::create(stencil));
    mpStencilFbo = Fbo::create(mpDevice);
}

ref<SVAO> SVAO::create(ref<Device> pDevice, const Properties& dict)
{
    auto pPass = make_ref<SVAO>(std::move(pDevice));
    for (const auto& [key, value] : dict)
    {
        if (key == kRadius) pPass->mData.radius = value;
        else if (key == kPrimaryDepthMode) pPass->mPrimaryDepthMode = value;
        else if (key == kSecondaryDepthMode) pPass->mSecondaryDepthMode = value;
        else if (key == kExponent) pPass->mData.exponent = value;
        else if (key == kUseRayPipeline) pPass->mUseRayPipeline = value;
        else if (key == kThickness) pPass->mData.thickness = value;
        else if (key == kImportanceThreshold) pPass->mData.importanceThreshold = value;
        else if (key == kTargetTime) pPass->mTargetTimeMs = value;
        else if (key == kStochMapDivisor) pPass->mStochMapDivisor = value;
        else if (key == kDualAo) pPass->mDualAo = value;
        else if (key == kAlphaTest) pPass->mAlphaTest = value;
        else logWarning("Unknown field '" + key + "' in a SVAO dictionary");
    }
    return pPass;
}

Properties SVAO::getProperties() const
{
    Properties d;
    d[kRadius] = mData.radius;
    d[kPrimaryDepthMode] = mPrimaryDepthMode;
    d[kSecondaryDepthMode] = mSecondaryDepthMode;
    d[kExponent] = mData.exponent;
    d[kUseRayPipeline] = mUseRayPipeline;
    d[kThickness] = mData.thickness;
    d[kImportanceThreshold] = mData.importanceThreshold;
    d[kTargetTime] = mTargetTimeMs;
    d[kStochMapDivisor] = mStochMapDivisor;
    d[kDualAo] = mDualAo;
    d[kAlphaTest] = mAlphaTest;
    return d;
}

RenderPassReflection SVAO::reflect(const CompileData& compileData)
{
    auto internalMapsRes = compileData.defaultTexDims;
    if(mStochMapDivisor > 1)
    {
        internalMapsRes.x = (internalMapsRes.x + mStochMapDivisor - 1) / mStochMapDivisor;
        internalMapsRes.y = (internalMapsRes.y + mStochMapDivisor - 1) / mStochMapDivisor;
    }

    RenderPassReflection reflector;
    //reflector.addInput(kAoStencil, "(Depth-) Stencil Buffer for the ao mask").format(ResourceFormat::D32FloatS8X24);
    reflector.addInput(kGbufferDepth, "Non-Linear Depth from the G-Buffer").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addInput(kDepth, "Linear Depth-buffer").bindFlags(ResourceBindFlags::ShaderResource)
        .texture2D(0, 0, 1, 0, 1); // allow mipmaps
    reflector.addInput(kDepth2, "Linear Depth-buffer of second layer").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addInput(kNormals, "World space normals, [0, 1] range").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addInput(kColor, "Color for pixel importance").bindFlags(ResourceBindFlags::ShaderResource);
    auto aoFormat = ResourceFormat::R8Unorm;
    if(mDualAo) aoFormat = ResourceFormat::RG8Unorm;
    reflector.addOutput(kAmbientMap, "Ambient Occlusion (bright/dark if dualAO is enabled)").bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::RenderTarget).format(aoFormat);
    reflector.addOutput(kAoStencil, "Stencil Bitmask for primary / secondary ao").bindFlags(ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource).format(ResourceFormat::R8Uint);
    //reflector.addInternal(kAoStencil2, "ping pong for stencil mask").bindFlags(ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource).format(ResourceFormat::R8Uint);
    //reflector.addInternal(kInternalStencil, "internal stencil mask").format(ResourceFormat::D32FloatS8X24);
    reflector.addOutput(kInternalStencil, "internal stencil mask").format(ResourceFormat::D32FloatS8X24).bindFlags(ResourceBindFlags::DepthStencil);

    reflector.addOutput(kInternalRayMin, "internal ray min").format(ResourceFormat::R32Int).bindFlags(ResourceBindFlags::AllColorViews).texture2D(internalMapsRes.x, internalMapsRes.y);
    reflector.addOutput(kInternalRayMax, "internal ray max").format(ResourceFormat::R32Int).bindFlags(ResourceBindFlags::AllColorViews).texture2D(internalMapsRes.x, internalMapsRes.y);

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
    Properties sdDict;
    sdDict["SampleCount"] = mStochSamples;
    sdDict["CullMode"] = RasterizerState::CullMode::Back;
    sdDict["AlphaTest"] = mAlphaTest;
    mpStochasticDepthGraph = RenderGraph::create(mpDevice, "Stochastic Depth");
    ref<RenderPass> pStochasticDepthPass;
    switch(mStochasticDepthImpl)
    {
    case StochasticDepthImpl::Raster:
        sdDict["linearize"] = true;
        sdDict["depthFormat"] = ResourceFormat::D32FloatS8X24;
        //sdDict["Alpha"] = 0.375f; // for 4 samples => ALPHA * 4 = 1.5 => 1.5 + rng will save 1-2 samples per pixel
        sdDict["Alpha"] = 1.5 / mStochSamples;
        pStochasticDepthPass = RenderPass::create("StochasticDepthMap", mpDevice, sdDict);
        break;
    case StochasticDepthImpl::Ray:
        sdDict["normalize"] = true;
        sdDict["depthFormat"] = ResourceFormat::R32Float;
        sdDict["useRayPipeline"] = true; // performs better than raster //mUseRayPipeline;
        pStochasticDepthPass = RenderPass::create("StochasticDepthMapRT", mpDevice, sdDict);    
        break;
    }
    mpStochasticDepthGraph->addPass(pStochasticDepthPass, "StochasticDepthMap");
    mpStochasticDepthGraph->markOutput("StochasticDepthMap.stochasticDepth");
    mpStochasticDepthGraph->setScene(mpScene);
    mStochLastSize = uint2(0);
}

void SVAO::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpScene) return;
    mFrameIndex++;

    auto pNonLinearDepth = renderData[kGbufferDepth]->asTexture();
    auto pDepth = renderData[kDepth]->asTexture();
    mHasDepthMipmap = pDepth->getMipCount() > 1;
    auto pNormal = renderData[kNormals]->asTexture();
    auto pAoDst = renderData[kAmbientMap]->asTexture();
    auto pDepth2 = renderData[kDepth2]->asTexture();
    auto pColor = renderData[kColor]->asTexture();

    auto pAoMask = renderData[kAoStencil]->asTexture();
    //auto pAoMask2 = renderData[kAoStencil2]->asTexture();
    auto pInternalStencil = renderData[kInternalStencil]->asTexture();

    auto pInternalRayMin = renderData[kInternalRayMin]->asTexture();
    auto pInternalRayMax = renderData[kInternalRayMax]->asTexture();

    if (!mEnabled)
    {
        pRenderContext->clearTexture(pAoDst.get(), float4(1.0f));
        return;
    }

    if (!mpRasterPass || !mpRasterPass2 || !mpRayProgram) // this needs to be deferred because it needs the scene defines to compile
    {
        // generate neural net shader files
        std::filesystem::path resPath;
        auto foundShader = findFileInShaderDirectories("RenderPasses/SVAO/SVAORaster.ps.slang", resPath);

        DefineList defines;
        defines.add("PRIMARY_DEPTH_MODE", std::to_string(uint32_t(mPrimaryDepthMode)));
        defines.add("SECONDARY_DEPTH_MODE", std::to_string(uint32_t(mSecondaryDepthMode)));
        defines.add("MSAA_SAMPLES", std::to_string(mStochSamples)); // TODO update this from gui
        defines.add("TRACE_OUT_OF_SCREEN", mTraceOutOfScreen ? "1" : "0");
        defines.add("STOCHASTIC_DEPTH_IMPL", std::to_string(uint32_t(mStochasticDepthImpl)));
        defines.add("USE_DEPTH_LOD", mUseDepthLod ? "1" : "0");
        defines.add("DEPTH_MIPS", std::to_string(mDepthTexMips));
        defines.add("STOCH_MAP_DIVISOR", std::to_string(mStochMapDivisor) + "u");
        defines.add("DUAL_AO", mDualAo ? "1" : "0");
        defines.add("USE_IMPORTANCE", mImportanceEnabled ? "1" : "0");
        defines.add("USE_ALPHA_TEST", mAlphaTest ? "1" : "0");
        defines.add(mpScene->getSceneDefines());

        // raster pass 1
        mpRasterPass = FullScreenPass::create(mpDevice, getFullscreenShaderDesc(kRasterShader), defines);

        // raster pass 2
        mpRasterPass2 = FullScreenPass::create(mpDevice, getFullscreenShaderDesc(kRasterShader2), defines);
        mpRasterPass2->getState()->setDepthStencilState(mpDepthStencilState);

        // ray pass
        RtProgram::Desc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kRayShader);
        desc.setMaxPayloadSize(kMaxPayloadSizePreventDarkHalos);
        desc.setMaxAttributeSize(mpScene->getRaytracingMaxAttributeSize());
        desc.setMaxTraceRecursionDepth(1);
        desc.addTypeConformances(mpScene->getTypeConformances());
        desc.setShaderModel("6_5");

        auto sbt = RtBindingTable::create(1, 1, mpScene->getGeometryCount());
        sbt->setRayGen(desc.addRayGen("rayGen"));
        sbt->setMiss(0, desc.addMiss("miss"));
        sbt->setHitGroup(0, mpScene->getGeometryIDs(GeometryType::TriangleMesh), desc.addHitGroup("closestHit", "anyHit"));
        // TODO add remaining primitives
        mpRayProgram = RtProgram::create(mpDevice, desc, defines);
        mRayVars = RtProgramVars::create(mpDevice, mpRayProgram, sbt);
        mDirty = true;
    }

    auto rasterVars = mpRasterPass->getRootVar();
    auto rasterVars2 = mpRasterPass2->getRootVar();
    auto rayVars = mRayVars->getRootVar();

    if (mDirty)
    {
        // update data raster
        rasterVars["StaticCB"].setBlob(mData);
        rasterVars["gNoiseSampler"] = mpNoiseSampler;
        rasterVars["gTextureSampler"] = mpTextureSampler;
        rasterVars["gNoiseTex"] = mpNoiseTexture;
        // update data raster 2
        rasterVars2["StaticCB"].setBlob(mData);
        rasterVars2["gNoiseSampler"] = mpNoiseSampler;
        rasterVars2["gTextureSampler"] = mpTextureSampler;
        rasterVars2["gNoiseTex"] = mpNoiseTexture;
        // update data ray
        rayVars["StaticCB"].setBlob(mData);
        rayVars["gNoiseSampler"] = mpNoiseSampler;
        rayVars["gTextureSampler"] = mpTextureSampler;
        rayVars["gNoiseTex"] = mpNoiseTexture;

        // also clear ao texture if guard band changed
        pRenderContext->clearTexture(pAoDst.get(), float4(0.0f));
        mDirty = false;
    }

    mpFbo->attachColorTarget(pAoDst, 0);
    mpFbo->attachColorTarget(pAoMask, 1);
    //mpFbo->attachColorTarget(mEnableRayFilter ? pAoMask2 : pAoMask, 1);

    auto pCamera = mpScene->getCamera().get();
    pCamera->setShaderData(rasterVars["PerFrameCB"]["gCamera"]);
    rasterVars["PerFrameCB"]["invViewMat"] = inverse(pCamera->getViewMatrix());
    if (mPrimaryDepthMode == DepthMode::PerfectClassify)
    {
        // set raytracing data
        auto var = mpRasterPass->getRootVar();
        mpScene->setRaytracingShaderData(pRenderContext, var);
    }
    rasterVars["PerFrameCB"]["useDepthLod"] = mUseDepthLod ? 1 : 0;
    rasterVars["PerFrameCB"]["frameIndex"] = mFrameIndex;
    rasterVars["PerFrameCB"]["pixelSkipX"] = mPixelSkipX;
    rasterVars["PerFrameCB"]["pixelSkipY"] = mPixelSkipY;

    setDepthTex(rasterVars, pDepth);
    rasterVars["gDepthTex2"] = pDepth2;
    rasterVars["gNormalTex"] = pNormal;
    rasterVars["gColor"] = pColor;
    //mpRasterPass["gDepthAccess"] = pAccessStencil;

    
    auto& dict = renderData.getDictionary();
    auto guardBand = dict.getValue("guardBand", 0);
    setGuardBandScissors(*mpRasterPass->getState(), renderData.getDefaultTextureDims(), guardBand);
    {
        FALCOR_PROFILE(pRenderContext, "AO 1");

        if (mSecondaryDepthMode == DepthMode::StochasticDepth)
        {
            FALCOR_PROFILE(pRenderContext, "Clear RayMinMax");
            pRenderContext->clearUAV(pInternalRayMin->getUAV().get(), uint4(asuint(std::numeric_limits<float>::max())));
            pRenderContext->clearUAV(pInternalRayMax->getUAV().get(), uint4(0u));
            rasterVars["gRayMinAccess"] = pInternalRayMin;
            rasterVars["gRayMaxAccess"] = pInternalRayMax;
        }

        mpRasterPass->execute(pRenderContext, mpFbo, false);
    }

    //if(mEnableRayFilter)
    //{
    //    FALCOR_PROFILE("RayFilter");
    //    mpRayFilter->execute(pRenderContext, pAoMask2, pAoMask); // result should be in pAoMask finally
    //}

    ref<Texture> pStochasticDepthMap;

    FALCOR_PROFILE(pRenderContext, "AORefine");

    auto profiler = pRenderContext->getProfiler();
    if (profiler && profiler->isEnabled())
    {
        auto event = profiler->findEvent("/onFrameRender/RenderGraphExe::execute()/SVAO/AORefine");
        if (event)
        {
            //auto timeMs = event->getGpuTimeAverage();
            auto timeMs = event->getGpuTime();
            mLastGpuTime = std::clamp(timeMs, mLastGpuTime * 0.99f, (mLastGpuTime + 0.01f) * 1.01f);

            if(mLastGpuTime > mTargetTimeMs * 1.05f)
            {
                // we spent too much time, increase the importance threshold
                mData.importanceThreshold = std::min(mData.importanceThreshold + 0.001f, 1.0f);
            }
            else if (mLastGpuTime < mTargetTimeMs * 0.7f)
            {
                // we can spend more time, lower the importance threshold
                mData.importanceThreshold = std::max(mData.importanceThreshold - 0.001f, 0.0f);
            }
            mDirty = true;
        }
    }

    //  execute stochastic depth map
    if (mSecondaryDepthMode == DepthMode::StochasticDepth)
    {
        switch (mStochasticDepthImpl)
        {
        case StochasticDepthImpl::Raster:
            mpStochasticDepthGraph->setInput("StochasticDepthMap.depthMap", pNonLinearDepth);
            break;
        case StochasticDepthImpl::Ray:
            mpStochasticDepthGraph->setInput("StochasticDepthMap.linearZ", pDepth);
            break;
        }
        mpStochasticDepthGraph->setInput("StochasticDepthMap.rayMin", pInternalRayMin);
        mpStochasticDepthGraph->setInput("StochasticDepthMap.rayMax", pInternalRayMax);
        //mpStochasticDepthGraph->setInput("StochasticDepthMap.stencilMask", pAccessStencil);
        auto stochSize = uint2(pAoDst->getWidth(), pAoDst->getHeight());
        if(mStochMapDivisor > 1u)
        {
            stochSize = (stochSize + uint2(mStochMapDivisor - 1)) / uint2(mStochMapDivisor); 
        }

        if(any(mStochLastSize != stochSize))
        {
            auto stochFbo = Fbo::create2D(mpDevice, stochSize.x, stochSize.y, ResourceFormat::R32Float);
            
            mpStochasticDepthGraph->onResize(stochFbo.get());
            mStochLastSize = stochSize;
        }
        
        mpStochasticDepthGraph->execute(pRenderContext);
        pStochasticDepthMap = mpStochasticDepthGraph->getOutput("StochasticDepthMap.stochasticDepth")->asTexture();
    }

    if (mUseRayPipeline) // RAY PIPELINE
    {
        // set raytracing data
        //mpScene->setRaytracingShaderData(pRenderContext, mRayVars);

        // set camera data
        pCamera->setShaderData(rayVars["PerFrameCB"]["gCamera"]);
        rayVars["PerFrameCB"]["invViewMat"] = inverse(pCamera->getViewMatrix());
        rayVars["PerFrameCB"]["guardBand"] = guardBand;

        // set textures
        setDepthTex(rayVars, pDepth);
        rayVars["gDepthTex2"] = pDepth2;
        rayVars["gNormalTex"] = pNormal;
        rayVars["gsDepthTex"] = pStochasticDepthMap;
        rayVars["aoMask"] = pAoMask;
        //mRayVars["aoPrev"] = pAoDst; // src view
        rayVars["output"] = pAoDst; // uav view


        uint3 dims = uint3(pAoDst->getWidth() - 2 * guardBand, pAoDst->getHeight() - 2 * guardBand, 1);
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
            mpStencilPass->getRootVar()["aoMask"] = pAoMask;
            mpStencilPass->execute(pRenderContext, mpStencilFbo);
            //pRenderContext->copySubresource(pInternalStencil.get(), 1, pAoMask.get(), 0); // <= don't do this, this results in a slow stencil
            
        }

        mpFbo2->attachDepthStencilTarget(pInternalStencil);
        mpFbo2->attachColorTarget(pAoDst, 0);

        // set raytracing data
        auto var = mpRasterPass2->getRootVar();
        mpScene->setRaytracingShaderData(pRenderContext, var);

        // set camera data
        pCamera->setShaderData(rasterVars2["PerFrameCB"]["gCamera"]);
        rasterVars2["PerFrameCB"]["invViewMat"] = inverse(pCamera->getViewMatrix());

        // set textures
        setDepthTex(rasterVars2, pDepth);
        rasterVars2["gDepthTex2"] = pDepth2;
        rasterVars2["gNormalTex"] = pNormal;
        rasterVars2["gsDepthTex"] = pStochasticDepthMap;
        rasterVars2["aoMask"] = pAoMask;
        rasterVars2["aoPrev"] = pAoDst;

        setGuardBandScissors(*mpRasterPass2->getState(), renderData.getDefaultTextureDims(), guardBand);

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
        //{ (uint32_t)DepthMode::MachineClassify, "MachineClassify" },
        //{ (uint32_t)DepthMode::MachinePredict, "MachinePredict" },
        //{ (uint32_t)DepthMode::PerfectClassify, "PerfectClassify" },
        {(uint32_t)DepthMode::Mipmaps, "Mipmaps"},
    };

    const Gui::DropdownList kSecondaryDepthModeDropdown =
    {
        { (uint32_t)DepthMode::StochasticDepth, "StochasticDepth" },
        { (uint32_t)DepthMode::Raytraced, "Raytraced" },
    };

    const Gui::DropdownList kStochasticDepthDopdown = {
        {(uint32_t)StochasticDepthImpl::Raster, "Raster"},
        {(uint32_t)StochasticDepthImpl::Ray, "Ray"},
    };

    const Gui::DropdownList kSampleCountList =
    {
        { (uint32_t)1, "1" },
        { (uint32_t)2, "2" },
        { (uint32_t)4, "4" },
        { (uint32_t)8, "8" },
        //{ (uint32_t)16, "16" }, // falcor (and directx) only support 8 render targets, which are required for the raster variant
    };


    auto reset = false;

    widget.checkbox("Enabled", mEnabled);
    if (!mEnabled) return;

    if (widget.checkbox("Alpha Test", mAlphaTest)) reset = true;

    uint32_t primaryDepthMode = (uint32_t)mPrimaryDepthMode;
    if (widget.dropdown("Primary Depth Mode", kPrimaryDepthModeDropdown, primaryDepthMode)) {
        mPrimaryDepthMode = (DepthMode)primaryDepthMode;
        reset = true;
    }

    if(widget.checkbox("Importance Driven", mImportanceEnabled)) reset = true;
    if(mImportanceEnabled)
    {
        if (widget.slider("Importance Threshold", mData.importanceThreshold, 0.0f, 1.0f)) mDirty = true;
        widget.var("Target Time", mTargetTimeMs, 0.0f, FLT_MAX, 0.1f);
        widget.text("current time: " + std::to_string(mLastGpuTime) + " ms");
    }
    

    uint32_t secondaryDepthMode = (uint32_t)mSecondaryDepthMode;
    if (widget.dropdown("Secondary Depth Mode", kSecondaryDepthModeDropdown, secondaryDepthMode)) {
        mSecondaryDepthMode = (DepthMode)secondaryDepthMode;
        reset = true;
    }

    if (mSecondaryDepthMode == DepthMode::StochasticDepth)
    {
        uint32_t stochasticImpl = (uint32_t)mStochasticDepthImpl;
        if (widget.dropdown("Stochastic Impl.", kStochasticDepthDopdown, stochasticImpl))
        {
            mStochasticDepthImpl = (StochasticDepthImpl)stochasticImpl;
            reset = true;
        }

        if (widget.dropdown("St. Sample Count", kSampleCountList, mStochSamples))
            reset = true;

        if(widget.var("SD-Map Divisor", mStochMapDivisor, 1u, 16u, 1u))
            reset = true;

        if (mpFbo->getWidth() % mStochMapDivisor != 0)
            widget.text("Warning: SD-Map Divisor does not divide width of screen");
        if (mpFbo->getHeight() % mStochMapDivisor != 0)
            widget.text("Warning: SD-Map Divisor does not divide height of screen");
    }

    if (widget.checkbox("Ray Pipeline", mUseRayPipeline)) reset = true;

    if (widget.var("Sample Radius", mData.radius, 0.01f, FLT_MAX, 0.01f)) mDirty = true;

    if (widget.var("Thickness", mData.thickness, 0.0f, 1.0f, 0.1f)) {
        mDirty = true;
        //mData.exponent = glm::mix(1.6f, 1.0f, mData.thickness);
    }

    if (widget.var("Power Exponent", mData.exponent, 1.0f, 4.0f, 0.1f)) mDirty = true;


    widget.separator();
    //if(mEnableRayFilter) mpRayFilter->renderUI(widget);

    if (widget.checkbox("Trace Out of Screen", mTraceOutOfScreen)) reset = true;
    widget.tooltip("If a sample point is outside of the screen, a ray is traced. Otherwise the closest sample from the border is used.");

    if (widget.var("Fade End (Screen Space Radius)", mData.ssRadiusFadeEnd, 0.0f, 100.0f, 1.0f)) mDirty = true;
    widget.tooltip("radius in pixels where the ray tracing result is completely faded and only rasterization remains");

    if (widget.var("Fade Size (Screen Space Radius)", mData.ssRadiusFadeSize, 0.01f, 100.0f, 1.0f)) mDirty = true;
    widget.tooltip("The fade will be for a screen space radius in range [FadeEnd, FadeEnd + FadeSize]");

    if(widget.checkbox("Output dual AO (bright/dark)", mDualAo)) reset = true;

    widget.checkbox("Use Depth LOD", mUseDepthLod);

    if (widget.var("Depth Mipmap Count", mDepthTexMips, 1u, 14u)) reset = true;

    widget.separator();

    widget.var("Pixel Skip X", mPixelSkipX, 1, 8);
    widget.var("Pixel Skip Y", mPixelSkipY, 1, 8);

    if (reset) requestRecompile();
}

void SVAO::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mpRasterPass.reset(); // new scene defines => recompile
    mpRasterPass2.reset();
    mpRayProgram.reset();
    if (mpStochasticDepthGraph)
        mpStochasticDepthGraph->setScene(pScene);
}

ref<Texture> SVAO::genNoiseTexture()
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

    return Texture::create2D(mpDevice, NOISE_SIZE, NOISE_SIZE, ResourceFormat::R8Unorm, 1, 1, data.data());
}

void SVAO::setDepthTex(ShaderVar& var, const ref<Texture>& pDepth)
{
    auto texArray = var["gDepthTexMips"];
    for(uint i = 0; i < mDepthTexMips; ++i)
    {
        if (i < pDepth->getMipCount())
            texArray[i].setSrv(pDepth->getSRV(i, 1));
        else
            texArray[i].setSrv(nullptr);
    }
}

Program::Desc SVAO::getFullscreenShaderDesc(const std::string& filename)
{
    Program::Desc desc;
    desc.addShaderModules(mpScene->getShaderModules());
    desc.addShaderLibrary(filename).psEntry("main");
    desc.addTypeConformances(mpScene->getTypeConformances());
    desc.setShaderModel("6_5");
    return desc;
}
