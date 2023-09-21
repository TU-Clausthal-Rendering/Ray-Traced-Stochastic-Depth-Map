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
#include "StochasticDepthMap.h"

namespace
{
    const std::string kDepthIn = "depthMap";
    const std::string ksDepth = "stochasticDepth";
    const std::string kInternalReservoirCounter = "internalReservoirCounter";
    const std::string kProgramFile = "RenderPasses/StochasticDepthMap/StochasticDepth.ps.slang";
    const std::string kStencilFile = "RenderPasses/StochasticDepthMap/Stencil.ps.slang";
    const std::string kSampleCount = "SampleCount";
    const std::string kAlpha = "Alpha";
    const std::string kReservoirSampling = "ReservoirSampling";
    const std::string kCullMode = "CullMode";
    const std::string kLinearize = "linearize";
    const std::string kDepthFormat = "depthFormat";
    const std::string kStencil = "stencilMask";
    const std::string kRayMin = "rayMin"; // ray T values for Ray.TMin
    const std::string kRayMax = "rayMax"; // ray T values for Ray.TMax
    const std::string kAlphaTest = "AlphaTest";

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
        { (uint32_t)16, "16" },
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
    registry.registerClass<RenderPass, StochasticDepthMap>();
}

StochasticDepthMap::StochasticDepthMap(ref<Device> pDevice) : RenderPass(std::move(pDevice))
{
    mpFbo = Fbo::create(mpDevice);

    DepthStencilState::Desc dsdesc;
    // set stencil test to pass when values > 0 are present (not equal to 0)
    dsdesc.setStencilEnabled(true);
    //dsdesc.setStencilWriteMask(0);
    //dsdesc.setStencilReadMask(1);
    dsdesc.setStencilOp(DepthStencilState::Face::FrontAndBack, DepthStencilState::StencilOp::Keep, DepthStencilState::StencilOp::Keep, DepthStencilState::StencilOp::Keep);
    dsdesc.setStencilFunc(DepthStencilState::Face::FrontAndBack, DepthStencilState::Func::NotEqual);
    dsdesc.setStencilRef(0);
    mpStencilState = DepthStencilState::create(dsdesc);


    mpStencilPass = FullScreenPass::create(mpDevice, kStencilFile);
    // modify stencil to write always a 1 if the depth test passes (pixel was not discarded)
    //dsdesc.setStencilWriteMask(0xff);
    dsdesc.setStencilOp(DepthStencilState::Face::FrontAndBack, DepthStencilState::StencilOp::Keep, DepthStencilState::StencilOp::Keep, DepthStencilState::StencilOp::Increase);
    dsdesc.setStencilFunc(DepthStencilState::Face::FrontAndBack, DepthStencilState::Func::Always);
    dsdesc.setStencilRef(1); // does not work currently, so op is set to increase
    // disable depth write
    dsdesc.setDepthEnabled(false);
    dsdesc.setDepthWriteMask(false);
    mpStencilPass->getState()->setDepthStencilState(DepthStencilState::create(dsdesc));
}

ref<StochasticDepthMap> StochasticDepthMap::create(ref<Device> pDevice, const Properties& dict)
{
    auto pPass = make_ref<StochasticDepthMap>(std::move(pDevice));
    for (const auto& [key, value] : dict)
    {
        if (key == kSampleCount) pPass->mSampleCount = value;
        else if (key == kAlpha) pPass->mAlpha = value;
        else if (key == kCullMode) pPass->mCullMode = value;
        else if (key == kLinearize) pPass->mLinearizeDepth = value;
        else if (key == kDepthFormat) pPass->mDepthFormat = value;
        else if (key == kReservoirSampling) pPass->mUseReservoirSampling = value;
        else if (key == kAlphaTest) pPass->mAlphaTest = value;
        else logWarning("Unknown field '" + key + "' in a StochasticDepthMap dictionary");
    }
    return pPass;
}

Properties StochasticDepthMap::getProperties() const
{
    Properties d;
    d[kSampleCount] = mSampleCount;
    d[kAlpha] = mAlpha;
    d[kCullMode] = mCullMode;
    d[kLinearize] = mLinearizeDepth;
    d[kDepthFormat] = mDepthFormat;
    d[kReservoirSampling] = mUseReservoirSampling;
    d[kAlphaTest] = mAlphaTest;
    return d;
}

RenderPassReflection StochasticDepthMap::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addInput(kDepthIn, "non-linear (primary) depth map").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addInput(kStencil, "(optional) stencil-mask").format(ResourceFormat::R8Uint).flags(RenderPassReflection::Field::Flags::Optional);
    reflector.addInput(kRayMin, "min ray T distance for depth values").flags(RenderPassReflection::Field::Flags::Optional);
    reflector.addInput(kRayMax, "max ray T distance for depth values").flags(RenderPassReflection::Field::Flags::Optional);
    reflector.addOutput(ksDepth, "stochastic depths in [0,1]").bindFlags(ResourceBindFlags::AllDepthViews).format(mDepthFormat).texture2D(0, 0, mSampleCount);

    reflector.addInternal(kInternalReservoirCounter, "reservoir counter").format(ResourceFormat::R32Uint).bindFlags(ResourceBindFlags::AllColorViews);
    return reflector;
}

void StochasticDepthMap::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    mpState.reset(); // reset defines etc.
    // force reload of camera cbuffer
    mLastZNear = 0.0f;
    mLastZFar = 0.0f;

    // always sample at pixel centers for our msaa resource
    static std::array<Fbo::SamplePosition, 16> samplePos = {};
    mpFbo->setSamplePositions(mSampleCount, 1, samplePos.data());

    // generate data for stratified sampling
    std::vector<int> indices;
    std::vector<uint32_t> lookUpTable;
    generateStratifiedLookupTable(mSampleCount, indices, lookUpTable);

    mpStratifiedIndices = Buffer::createStructured(mpDevice, sizeof(indices[0]), uint32_t(indices.size()), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, indices.data(), false);
    mpStratifiedLookUpBuffer = Buffer::createStructured(mpDevice, sizeof(lookUpTable[0]), uint32_t(lookUpTable.size()), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, lookUpTable.data(), false);
}

void StochasticDepthMap::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpScene) return;

    auto pDepthIn = renderData[kDepthIn]->asTexture();
    auto psDepths = renderData[ksDepth]->asTexture();
    ref<Texture> pStencilMask;
    if (renderData[kStencil]) pStencilMask = renderData[kStencil]->asTexture();
    if (!pStencilMask && renderData[kRayMax]) pStencilMask = renderData[kRayMax]->asTexture();
        
    if(pStencilMask && !isStencilFormat(mDepthFormat))
    {
        logWarning("StochasticDepthMap depth format must have stencil to enable writing to stencil");
        pStencilMask.reset();
    }
    ref<Texture> pRayMin;
    if (renderData[kRayMin]) pRayMin = renderData[kRayMin]->asTexture();
    ref<Texture> pRayMax;
    if (renderData[kRayMax]) pRayMax = renderData[kRayMax]->asTexture();

    ref<Texture> pReservoirCounter = renderData[kInternalReservoirCounter]->asTexture();
    
    auto pCamera = mpScene->getCamera();

    // clear
    pRenderContext->clearDsv(psDepths->getDSV().get(), 1.0f, 0, true, pStencilMask != nullptr);
    mpFbo->attachDepthStencilTarget(psDepths);

    if(!mpState)
    {
        Program::Desc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kProgramFile).vsEntry("vsMain").psEntry("psMain");
        desc.addTypeConformances(mpScene->getTypeConformances());
        desc.setShaderModel("6_2");
        auto defines = mpScene->getSceneDefines();
        defines.add("NUM_SAMPLES", std::to_string(mSampleCount));
        defines.add("ALPHA", std::to_string(mAlpha));
        defines.add("RESERVOIR_SAMPLING", mUseReservoirSampling ? "1" : "0");
        defines.add("INV_RESOLUTION", "float2(" + std::to_string(1.0f / mpFbo->getWidth()) + ", " + std::to_string(1.0f / mpFbo->getHeight()) + ")");
        defines.add("USE_ALPHA_TEST", mAlphaTest ? "1" : "0");
        if (mLinearizeDepth) defines.add("LINEARIZE");
        auto pProgram = GraphicsProgram::create(mpDevice, desc, defines);

        mpState = GraphicsState::create(mpDevice);
        mpState->setProgram(pProgram);
        mpVars = GraphicsVars::create(mpDevice, pProgram->getReflector());
        auto vars = mpVars->getRootVar();
        // linear sampler for downsampling depth buffer (if half res)
        vars["S"] = Sampler::create(mpDevice, Sampler::Desc().setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear));
    }

    if (pStencilMask)
    {
        FALCOR_PROFILE(pRenderContext, "Stencil Copy");
        mpStencilPass->getRootVar()["mask"] = pStencilMask;
        mpStencilPass->execute(pRenderContext, mpFbo);

        mpState->setDepthStencilState(mpStencilState);
    }
    else mpState->setDepthStencilState(nullptr);

    {
        FALCOR_PROFILE(pRenderContext, "Stochastic Depths");

        // rasterize non-linear depths
        mpState->setFbo(mpFbo);
        auto var = mpVars->getRootVar();
        var["stratifiedIndices"] = mpStratifiedIndices;
        var["stratifiedLookUpTable"] = mpStratifiedLookUpBuffer;
        var["depthBuffer"] = pDepthIn;
        var["rayMin"] = pRayMin;
        var["rayMax"] = pRayMax;

        if (mUseReservoirSampling)
        {
            pRenderContext->clearUAV(pReservoirCounter->getUAV().get(), uint4(0));
            var["counter"] = pReservoirCounter;
        }

        // set camera data
        float zNear = pCamera->getNearPlane();
        float zFar = pCamera->getFarPlane();
        if (mLastZNear != zNear || mLastZFar != zFar)
        {
            var["CameraCB"]["zNear"] = zNear;
            var["CameraCB"]["zFar"] = zFar;
            mLastZNear = zNear;
            mLastZFar = zFar;
        }

        mpScene->rasterize(pRenderContext, mpState.get(), mpVars.get(), mCullMode);
    }
}

void StochasticDepthMap::renderUI(Gui::Widgets& widget)
{
    static const Gui::DropdownList kDepthFormats =
    {
        { (uint32_t)ResourceFormat::D16Unorm, "D16Unorm"},
        { (uint32_t)ResourceFormat::D32Float, "D32Float" },
        { (uint32_t)ResourceFormat::D24UnormS8, "D24UnormS8" },
        //{ (uint32_t)ResourceFormat::D32FloatS8X24, "D32FloatS8X24" },
    };

    if (widget.checkbox("Alpha Test", mAlphaTest))
        requestRecompile();

    uint32_t depthFormat = (uint32_t)mDepthFormat;
    if (widget.dropdown("Buffer Format", kDepthFormats, depthFormat))
    {
        mDepthFormat = ResourceFormat(depthFormat);
        requestRecompile();
    }

    uint32_t cullMode = (uint32_t)mCullMode;
    if (widget.dropdown("Cull mode", kCullModeList, cullMode))
        mCullMode = (RasterizerState::CullMode)cullMode;

    if (widget.dropdown("Sample Count", kSampleCountList, mSampleCount))
        requestRecompile(); // reload pass (recreate texture)

    if (widget.var("Alpha", mAlpha, 0.0f, 1.0f, 0.01f))
        requestRecompile();

    if (widget.checkbox("Linearize Depths", mLinearizeDepth))
        requestRecompile();
}

void StochasticDepthMap::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mpState.reset();

    // force reload of camera cbuffer
    mLastZNear = 0.0f;
    mLastZFar = 0.0f;
}
