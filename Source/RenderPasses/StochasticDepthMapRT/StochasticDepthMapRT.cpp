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
#include "StochasticDepthMapRT.h"

namespace 
{
    const std::string kDepthIn = "linearZ";
    const std::string ksDepth = "stochasticDepth";
    const std::string kRayMin = "rayMin"; // ray T values for Ray.TMin
    const std::string kRayMax = "rayMax"; // ray T values for Ray.TMax
    const std::string kStencil = "stencilMask";

    const std::string kInternalStencil = "internalStencil";

    const std::string kRayShader = "RenderPasses/StochasticDepthMapRT/StochasticDepthMapRT.rt.slang";
    const std::string kRasterShader = "RenderPasses/StochasticDepthMapRT/StochasticDepthMapRT.ps.slang";
    const std::string kStencilFile = "RenderPasses/StochasticDepthMapRT/Stencil.ps.slang";
    
    const std::string kSampleCount = "SampleCount";
    const std::string kCullMode = "CullMode";
    const std::string kNormalize = "normalize";
    const std::string kUseRayPipeline = "useRayPipeline";
    const std::string kAlphaTest = "AlphaTest";
    const std::string kUse16Bit = "Use16Bit";
    const std::string kStoreNormals = "StoreNormals";
    const std::string kJitter = "Jitter";
    const std::string kImplementation = "Implementation";
    const std::string kAlpha = "Alpha";
    const std::string kRayInterval = "RayInterval";

    const Gui::DropdownList kCullModeList =
    {
        { (uint32_t)RasterizerState::CullMode::None, "None" },
        { (uint32_t)RasterizerState::CullMode::Back, "Back" },
        { (uint32_t)RasterizerState::CullMode::Front, "Front" },
    };

    const Gui::DropdownList kSampleCountList =
    {
        { (uint32_t)1, "1" },
        { (uint32_t)2, "2" },
        { (uint32_t)4, "4" },
        { (uint32_t)8, "8" },
        //{ (uint32_t)16, "16" }, // falcor (and directx) only support 8 render targets, which are required for the raster variant
    };
}

// code from stochastic-depth ambient occlusion:

// Based on Pascal's Triangle, 
// from https://www.geeksforgeeks.org/binomial-coefficient-dp-9/
static int Binomial(int n, int k)
{
    std::vector<int> C(k + 1);
    C[0] = 1;  // nC0 is 1 

    for (int i = 1; i <= n; i++)
    {
        // Compute next row of pascal triangle using 
        // the previous row 
        for (int j = std::min(i, k); j > 0; j--)
            C[j] = C[j] + C[j - 1];
    }
    return C[k];
}

static uint8_t count_bits(uint32_t v)
{
    uint8_t bits = 0;
    for (; v; ++bits) { v &= v - 1; }
    return bits;
}

void generateStratifiedLookupTable(int n, std::vector<int>& indices, std::vector<uint32_t>& lookUpTable) {

    uint32_t maxEntries = uint32_t(std::pow(2, n));
    //std::vector<int> indices(n + 1);
    //std::vector<uint32_t> lookUpTable(maxEntries);
    indices.resize(n + 1);
    lookUpTable.resize(maxEntries);

    // Generate index list
    indices[0] = 0;
    for (int i = 1; i <= n; i++) {
        indices[i] = Binomial(n, i - 1) + indices[i - 1];
    }

    // Generate lookup table
    std::vector<int> currentIndices(indices);
    lookUpTable[0] = 0;
    for (uint32_t i = 1; i < maxEntries; i++) {
        int popCount = count_bits(i);
        int index = currentIndices[popCount];
        lookUpTable[index] = i;
        currentIndices[popCount]++;
    }
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, StochasticDepthMapRT>();
}


StochasticDepthMapRT::StochasticDepthMapRT(ref<Device> pDevice) : RenderPass(std::move(pDevice))
{
    DepthStencilState::Desc dsdesc;
    // set stencil test to pass when values > 0 are present (not equal to 0)
    dsdesc.setStencilEnabled(true);
    //dsdesc.setStencilWriteMask(0);
    //dsdesc.setStencilReadMask(1);
    dsdesc.setStencilOp(DepthStencilState::Face::FrontAndBack, DepthStencilState::StencilOp::Keep, DepthStencilState::StencilOp::Keep, DepthStencilState::StencilOp::Keep);
    dsdesc.setStencilFunc(DepthStencilState::Face::FrontAndBack, DepthStencilState::Func::NotEqual);
    dsdesc.setStencilRef(0);
    // disable depth read/write
    dsdesc.setDepthEnabled(false);
    dsdesc.setDepthWriteMask(false);
    mpStencilState = DepthStencilState::create(dsdesc);


    mpStencilPass = FullScreenPass::create(mpDevice, kStencilFile);
    // modify stencil to write always a 1 if the depth test passes (pixel was not discarded)
    //dsdesc.setStencilWriteMask(0xff);
    dsdesc.setStencilOp(DepthStencilState::Face::FrontAndBack, DepthStencilState::StencilOp::Keep, DepthStencilState::StencilOp::Keep, DepthStencilState::StencilOp::Increase);
    dsdesc.setStencilFunc(DepthStencilState::Face::FrontAndBack, DepthStencilState::Func::Always);
    dsdesc.setStencilRef(1); // does not work currently, so op is set to increase
    mpStencilPass->getState()->setDepthStencilState(DepthStencilState::create(dsdesc));
}

ref<StochasticDepthMapRT> StochasticDepthMapRT::create(ref<Device> pDevice, const Properties& dict)
{
    auto pPass = make_ref<StochasticDepthMapRT>(std::move(pDevice));
    for (const auto& [key, value] : dict)
    {
        if (key == kSampleCount) pPass->mSampleCount = value;
        else if (key == kCullMode) pPass->mCullMode = value;
        else if (key == kNormalize) pPass->mNormalize = value;
        else if (key == kUseRayPipeline) pPass->mUseRayPipeline = value;
        else if (key == kAlphaTest) pPass->mAlphaTest = value;
        else if (key == kUse16Bit) pPass->mUse16Bit = value;
        else if (key == kStoreNormals) pPass->mStoreNormals = value;
        else if (key == kJitter) pPass->mJitter = value;
        else if (key == kImplementation) pPass->mImplementation = value;
        else if (key == kAlpha) pPass->mAlpha = value;
        else if (key == kRayInterval) pPass->mUseRayInterval = value;
        else logWarning("Unknown field '" + key + "' in a StochasticDepthMapRT dictionary");
    }
    return pPass;
}

Properties StochasticDepthMapRT::getProperties() const
{
    Properties d;
    d[kSampleCount] = mSampleCount;
    d[kCullMode] = mCullMode;
    d[kNormalize] = mNormalize;
    d[kAlphaTest] = mAlphaTest;
    //d[kUseRayPipeline] = mUseRayPipeline;
    d[kUse16Bit] = mUse16Bit;
    d[kStoreNormals] = mStoreNormals;
    d[kJitter] = mJitter;
    d[kImplementation] = mImplementation;
    d[kAlpha] = mAlpha;
    return d;
}

RenderPassReflection StochasticDepthMapRT::reflect(const CompileData& compileData)
{
    auto depthFormat = ResourceFormat::R32Float;
    if(mSampleCount == 2)
        depthFormat = ResourceFormat::RG32Float;
    else if(mSampleCount == 4)
        depthFormat = ResourceFormat::RGBA32Float;
    else throw std::runtime_error("StochasticDepthMapRT: Only 1, 2 and 4 samples are supported");

    if (mUse16Bit)
    {
        depthFormat = ResourceFormat::R16Float;
        if (mSampleCount == 2)
            depthFormat = ResourceFormat::RG16Float;
        else if (mSampleCount == 4)
            depthFormat = ResourceFormat::RGBA16Float;
        else throw std::runtime_error("StochasticDepthMapRT: Only 1, 2 and 4 samples are supported");
    }
    if(mStoreNormals)
    {
        throw std::runtime_error("StochasticDepthMapRT: Storing normals is not supported yet");
        //depthFormat = ResourceFormat::RG32Float;
        //if (mUse16Bit) depthFormat = ResourceFormat::RG16Uint; // TODO
    }

    RenderPassReflection reflector;
    reflector.addInput(kDepthIn, "non-linear (primary) depth map").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addInput(kStencil, "(optional) stencil-mask").format(ResourceFormat::R8Uint).flags(RenderPassReflection::Field::Flags::Optional);
    reflector.addInput(kRayMin, "min ray T distance for depth values").flags(RenderPassReflection::Field::Flags::Optional);
    reflector.addInput(kRayMax, "max ray T distance for depth values").flags(RenderPassReflection::Field::Flags::Optional);
    reflector.addOutput(ksDepth, "stochastic depths in [0,1]").bindFlags(ResourceBindFlags::AllColorViews).format(depthFormat).texture2D(0, 0, 1, 1, 1);
    reflector.addInternal(kInternalStencil, "stencil-mask").bindFlags(ResourceBindFlags::DepthStencil).format(ResourceFormat::D32FloatS8X24);
    return reflector;
}

void StochasticDepthMapRT::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    mpRayProgram.reset();
    mpRasterProgram.reset();
    mpFbo = Fbo::create(mpDevice);

    // generate data for stratified sampling
    std::vector<int> indices;
    std::vector<uint32_t> lookUpTable;
    generateStratifiedLookupTable(mSampleCount, indices, lookUpTable);

    mpStratifiedIndices = Buffer::createStructured(mpDevice, sizeof(indices[0]), uint32_t(indices.size()), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, indices.data(), false);
    mpStratifiedLookUpBuffer = Buffer::createStructured(mpDevice, sizeof(lookUpTable[0]), uint32_t(lookUpTable.size()), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, lookUpTable.data(), false);
}

void StochasticDepthMapRT::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpScene) return;

    auto pDepthIn = renderData[kDepthIn]->asTexture();
    auto psDepths = renderData[ksDepth]->asTexture();
    ref<Texture> pStencilMask;
    if (renderData[kStencil]) pStencilMask = renderData[kStencil]->asTexture();
    if (!pStencilMask && renderData[kRayMax]) pStencilMask = renderData[kRayMax]->asTexture();

    ref<Texture> pRayMin;
    if (renderData[kRayMin]) pRayMin = renderData[kRayMin]->asTexture();
    ref<Texture> pRayMax;
    if (renderData[kRayMax]) pRayMax = renderData[kRayMax]->asTexture();

    if(mClear)
    {
        pRenderContext->clearTexture(psDepths.get());
        mClear = false;
    }
    
#ifdef _DEBUG
    pRenderContext->clearTexture(psDepths.get()); // for debug, clear the texture first to better see what is written
#endif
    
    //if(!mpRayProgram || !mpRasterProgram)
    if (!mpRayProgram)
    {
        auto defines = mpScene->getSceneDefines();
        defines.add("NUM_SAMPLES", std::to_string(mSampleCount));
        defines.add("ALPHA", std::to_string(mAlpha));
        defines.add("NORMALIZE", mNormalize ? "1" : "0");
        auto rayConeSpread = mpScene->getCamera()->computeScreenSpacePixelSpreadAngle(renderData.getDefaultTextureDims().y);
        defines.add("RAY_CONE_SPREAD", std::to_string(rayConeSpread));
        defines.add("USE_ALPHA_TEST", mAlphaTest ? "1" : "0");
        defines.add("sd_t", mStoreNormals ? "float2" : "float");
        defines.add("SD_USE_NORMALS", mStoreNormals ? "1" : "0");
        defines.add("SD_JITTER", mJitter ? "1" : "0");
        defines.add("IMPLEMENTATION", std::to_string(uint32_t(mImplementation)));
        defines.add("USE_RAY_INTERVAL", mUseRayInterval ? "1" : "0");

        // raster pass
        /* {
            Program::Desc desc;
            desc.addShaderModules(mpScene->getShaderModules());
            desc.addShaderLibrary(kRasterShader).psEntry("main");
            desc.addTypeConformances(mpScene->getTypeConformances());
            desc.setShaderModel("6_5");

            mpRasterProgram = FullScreenPass::create(mpDevice, desc, defines);
            auto vars = mpRasterProgram->getRootVar();
            vars["S"] = Sampler::create(mpDevice, Sampler::Desc().setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear));

            mpRasterProgram->getRootVar()["stratifiedIndices"] = mpStratifiedIndices;
            mpRasterProgram->getRootVar()["stratifiedLookUpTable"] = mpStratifiedLookUpBuffer;
        }*/

        // ray pass
        {
            RtProgram::Desc desc;
            desc.addShaderModules(mpScene->getShaderModules());
            desc.addShaderLibrary(kRayShader);
            uint32_t normalsCount = mStoreNormals ? mSampleCount : 0;
            desc.setMaxPayloadSize((mSampleCount + normalsCount + 1) * sizeof(float));
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
            auto vars = mRayVars->getRootVar();
            vars["S"] = Sampler::create(mpDevice, Sampler::Desc().setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear));

            mRayVars->getRootVar()["stratifiedIndices"] = mpStratifiedIndices;
            mRayVars->getRootVar()["stratifiedLookUpTable"] = mpStratifiedLookUpBuffer;
        }
    }

    if (mUseRayPipeline)
    {
        mRayVars->getRootVar()["depthInTex"] = pDepthIn;
        mRayVars->getRootVar()["depthOutTex"] = psDepths;
        mRayVars->getRootVar()["maskTex"] = pStencilMask;
        mRayVars->getRootVar()["rayMinTex"] = pRayMin;
        mRayVars->getRootVar()["rayMaxTex"] = pRayMax;
        mRayVars->getRootVar()["materialAlphaTestLookup"] = mpMaterialAlphaTest;

        mpScene->raytrace(pRenderContext, mpRayProgram.get(), mRayVars, uint3(psDepths->getWidth(), psDepths->getHeight(), 1));
    
    }
    else // raster pipeline
    {
        assert(false);
        __debugbreak();
        return; // needs to be updated!!:
        if(pStencilMask)
        {
            FALCOR_PROFILE(pRenderContext, "Stencil Clear&Copy");

            // clear stencil buffer and attach to fbo
            auto stencil = renderData[kInternalStencil]->asTexture();
            pRenderContext->clearDsv(stencil->getDSV().get(), 1.0f, 0, false, true);
            mpFbo->attachDepthStencilTarget(stencil, 0, 0, 1);

            // copy mask to stencil buffer
            mpStencilPass->getRootVar()["mask"] = pStencilMask;
            mpStencilPass->execute(pRenderContext, mpFbo);

            // set state for raster pass
            mpRasterProgram->getState()->setDepthStencilState(mpStencilState);
        }
        else
        {
            mpRasterProgram->getState()->setDepthStencilState(nullptr);
        }

        mpRasterProgram->getRootVar()["depthInTex"] = pDepthIn;
        mpRasterProgram->getRootVar()["rayMinTex"] = pRayMin;
        mpRasterProgram->getRootVar()["rayMaxTex"] = pRayMax;
        mpRasterProgram->getRootVar()["materialAlphaTestLookup"] = mpMaterialAlphaTest;

        // set gScene and raytracing data
        mpScene->setRaytracingShaderData(pRenderContext, mpRasterProgram->getRootVar());

        for (uint i = 0; i < mSampleCount; ++i)
        {
            mpFbo->attachColorTarget(psDepths, i, 0, i, 1);
            //auto rtv = psDepths->getRTV(0, i, 1);
            
        }
        FALCOR_PROFILE(pRenderContext, "Raster");
        mpRasterProgram->execute(pRenderContext, mpFbo);
    }
}

void StochasticDepthMapRT::renderUI(Gui::Widgets& widget)
{
    widget.button("Clear", mClear);

    if (widget.dropdown("Sample Count", kSampleCountList, mSampleCount))
        requestRecompile(); // reload pass (recreate texture)

    if (widget.checkbox("Normalize Depths", mNormalize))
        requestRecompile();

    if (widget.checkbox("Alpha Test", mAlphaTest))
        requestRecompile();

    // select implementation
    if (widget.dropdown("Implementation", mImplementation))
        requestRecompile();

    //if (widget.checkbox("Use Ray Pipeline", mUseRayPipeline))
    //    requestRecompile();
}

void StochasticDepthMapRT::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    // recompile shaders
    mpRayProgram.reset();
    mpRasterProgram.reset();
    mpMaterialAlphaTest.reset();

    if(mpScene)
    {
        // fill mpMateralAlphaTest buffer with info about alpha tested materials
        auto bufferSize = (mpScene->getMaterialCount() + 31) / 32;
        bufferSize = std::max(bufferSize, 1u); // make sure we have at least one dword
        std::vector<uint32_t> alphaTested(bufferSize, 0u);
        for(size_t i = 0; i < mpScene->getMaterialCount(); i++)
        {
            auto dword = i / 32u;
            auto bit = i % 32u;
            if(mpScene->getMaterial(MaterialID(i))->getAlphaMode() == AlphaMode::Mask)
                alphaTested[dword] |= 1u << bit; // set bit
        }

        //mpMaterialAlphaTest = Buffer::create(mpDevice, bufferSize * sizeof(uint32_t), Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, alphaTested.data());
        mpMaterialAlphaTest = Buffer::createStructured(mpDevice, sizeof(uint32_t), bufferSize, Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, alphaTested.data(), false);
    }
}

