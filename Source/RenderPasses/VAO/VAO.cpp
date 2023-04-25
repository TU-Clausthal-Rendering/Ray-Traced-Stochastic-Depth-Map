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
#include "VAO.h"
#include "../Utils/GuardBand/guardband.h"
#include "glm/gtc/random.hpp"
#include "Utils/Math/FalcorMath.h"

namespace
{
    const Gui::DropdownList kDistributionDropdown =
    {
        { (uint32_t)VAO::SampleDistribution::Random, "Random" },
        { (uint32_t)VAO::SampleDistribution::VanDerCorput, "Uniform VanDerCorput" },
        { (uint32_t)VAO::SampleDistribution::Poisson, "Poisson" },
        { (uint32_t)VAO::SampleDistribution::Triangle, "Triangle" },
    };

    const Gui::DropdownList kDepthModeDropdown =
    {
        { (uint32_t)DepthMode::SingleDepth, "SingleDepth" },
        { (uint32_t)DepthMode::DualDepth, "DualDepth" },
        { (uint32_t)DepthMode::StochasticDepth, "StochasticDepth" },
        { (uint32_t)DepthMode::Raytraced, "Raytraced" }
    };

    const std::string kEnabled = "enabled";
    const std::string kKernelSize = "kernelSize";
    const std::string kDistribution = "distribution";
    const std::string kRadius = "radius";
    const std::string kDepthMode = "depthMode";
    const std::string kGuardBand = "guardBand";
    const std::string kThickness = "thickness";

    const std::string kAmbientMap = "ambientMap";
    const std::string kDepth = "depth";
    const std::string kDepth2 = "depth2";
    const std::string ksDepth = "stochasticDepth";
    const std::string kNormals = "normals";
    const std::string kMaterial = "materialData";
    //const std::string kInstanceID = "instanceID";

    const std::string kInternalRasterDepth = "iRasterDepth";
    const std::string kInternalRayDepth = "iRayDepth";
    const std::string kInternalAskRay = "iAskRay"; // needs ray tracing?
    const std::string kInternalRequireRay = "iRequireRay"; // answer to ray tracing
    const std::string kInternalForceRay = "iForceRay"; // force ray (or non ray) behaviour for out of screen or double sided
    //const std::string kInternalInstanceID = "iInstanceID";
    const std::string kInternalRasterAO = "iRasterAO";
    const std::string kInternalRayAO = "iRayAO";
    const std::string kInternalSphereEnd = "iSphereEnd";

    const std::string kSSAOShader = "RenderPasses/VAO/VAO.ps.slang";

    static const int NOISE_SIZE = 4; // in each dimension: 4x4
}

static void regVAO(pybind11::module& m)
{
    pybind11::class_<VAO, RenderPass, VAO::SharedPtr> pass(m, "VAO");
    pass.def("saveDepths", &VAO::saveDepths);

    pybind11::enum_<VAO::SampleDistribution> sampleDistribution(m, "SampleDistribution");
    sampleDistribution.value("Random", VAO::SampleDistribution::Random);
    sampleDistribution.value("VanDerCorput", VAO::SampleDistribution::VanDerCorput);
    sampleDistribution.value("Poisson", VAO::SampleDistribution::Poisson);

    pybind11::enum_<Falcor::DepthMode> depthMode(m, "DepthMode");
    depthMode.value("SingleDepth", Falcor::DepthMode::SingleDepth);
    depthMode.value("DualDepth", Falcor::DepthMode::DualDepth);
    depthMode.value("StochasticDepth", Falcor::DepthMode::StochasticDepth);
    depthMode.value("Raytraced", Falcor::DepthMode::Raytraced);
    depthMode.value("MachineClassify", Falcor::DepthMode::MachineClassify);
    depthMode.value("MachinePredict", Falcor::DepthMode::MachinePredict);
    depthMode.value("PerfectClassify", Falcor::DepthMode::PerfectClassify);
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, VAO>();
    ScriptBindings::registerBinding(regVAO);
}

VAO::VAO(std::shared_ptr<Device> pDevice) : RenderPass(std::move(pDevice))
{
    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(Sampler::Filter::Point, Sampler::Filter::Point, Sampler::Filter::Point).setAddressingMode(Sampler::AddressMode::Wrap, Sampler::AddressMode::Wrap, Sampler::AddressMode::Wrap);
    mpNoiseSampler = Sampler::create(mpDevice.get(), samplerDesc);

    samplerDesc.setFilterMode(Sampler::Filter::Point, Sampler::Filter::Point, Sampler::Filter::Point).setAddressingMode(Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp);
    mpTextureSampler = Sampler::create(mpDevice.get(), samplerDesc);

    mpAOFbo = Fbo::create(mpDevice.get());
}

VAO::SharedPtr VAO::create(std::shared_ptr<Device> pDevice, const Dictionary& dict)
{
    SharedPtr pPass = SharedPtr(new VAO(std::move(pDevice)));
    for (const auto& [key, value] : dict)
    {
        if (key == kEnabled) pPass->mEnabled = value;
        else if (key == kKernelSize) pPass->mKernelSize = value;
        else if (key == kDistribution) pPass->mHemisphereDistribution = value;
        else if (key == kRadius) pPass->mData.radius = value;
        else if (key == kDepthMode) pPass->mDepthMode = value;
        else if (key == kGuardBand) pPass->mGuardBand = value;
        else if (key == kThickness) pPass->mData.thickness = value;
        else logWarning("Unknown field '" + key + "' in a VAO dictionary");
    }
    return pPass;
}

Dictionary VAO::getScriptingDictionary()
{
    Dictionary dict;
    dict[kEnabled] = mEnabled;
    dict[kKernelSize] = mKernelSize;
    dict[kRadius] = mData.radius;
    dict[kDistribution] = mHemisphereDistribution;
    dict[kDepthMode] = mDepthMode;
    dict[kGuardBand] = mGuardBand;
    dict[kThickness] = mData.thickness;
    return dict;
}

RenderPassReflection VAO::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addInput(kDepth, "Linear Depth-buffer").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addInput(kDepth2, "Linear Depth-buffer of second layer").bindFlags(ResourceBindFlags::ShaderResource).flags(RenderPassReflection::Field::Flags::Optional);
    reflector.addInput(kNormals, "World space normals, [0, 1] range").bindFlags(ResourceBindFlags::ShaderResource);
    //reflector.addInput(kInstanceID, "Instance ID").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addInput(ksDepth, "Linear Stochastic Depth Map").texture2D(0, 0, 0).bindFlags(ResourceBindFlags::ShaderResource).flags(RenderPassReflection::Field::Flags::Optional);
    reflector.addInput(kMaterial, "Material data").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addOutput(kAmbientMap, "Ambient Occlusion").bindFlags(Falcor::ResourceBindFlags::RenderTarget).format(ResourceFormat::R8Unorm);

    reflector.addInternal(kInternalRasterDepth, "internal raster depth").texture2D(0, 0, 1, 1, mKernelSize)
        .bindFlags(ResourceBindFlags::UnorderedAccess).format(ResourceFormat::R32Float);
    reflector.addInternal(kInternalRayDepth, "internal raster depth").texture2D(0, 0, 1, 1, mKernelSize)
        .bindFlags(ResourceBindFlags::UnorderedAccess).format(ResourceFormat::R32Float);
    reflector.addInternal(kInternalAskRay, "internal ask ray").texture2D(0, 0, 1, 1, mKernelSize)
        .bindFlags(ResourceBindFlags::UnorderedAccess).format(ResourceFormat::R8Uint);
    reflector.addInternal(kInternalRequireRay, "internal require ray").texture2D(0, 0, 1, 1, mKernelSize)
        .bindFlags(ResourceBindFlags::UnorderedAccess).format(ResourceFormat::R8Uint);
    reflector.addInternal(kInternalForceRay, "internal force ray").texture2D(0, 0, 1, 1, mKernelSize)
        .bindFlags(ResourceBindFlags::UnorderedAccess).format(ResourceFormat::R8Uint);
    //reflector.addInternal(kInternalInstanceID, "internal instance ID").texture2D(0, 0, 1, 1, mKernelSize)
    //    .bindFlags(ResourceBindFlags::UnorderedAccess).format(ResourceFormat::R8Uint);
    reflector.addInternal(kInternalRasterAO, "internal raster AO").texture2D(0, 0, 1, 1, mKernelSize)
        .bindFlags(ResourceBindFlags::UnorderedAccess).format(ResourceFormat::R8Unorm);
    reflector.addInternal(kInternalRayAO, "internal ray AO").texture2D(0, 0, 1, 1, mKernelSize)
        .bindFlags(ResourceBindFlags::UnorderedAccess).format(ResourceFormat::R8Unorm);
    reflector.addInternal(kInternalSphereEnd, "internal sphere end").texture2D(0, 0, 1, 1, mKernelSize)
        .bindFlags(ResourceBindFlags::UnorderedAccess).format(ResourceFormat::R32Float);

    return reflector;
}

void VAO::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    setKernel();
    setNoiseTexture();

    mDirty = true; // texture size may have changed => reupload data
    mpSSAOPass.reset();
}

void VAO::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpScene) return;

    auto pDepth = renderData[kDepth]->asTexture();
    auto pNormals = renderData[kNormals]->asTexture();
    auto pAoDst = renderData[kAmbientMap]->asTexture();
    //auto pInstanceID = renderData[kInstanceID]->asTexture();
    Texture::SharedPtr pDepth2;
    if (renderData[kDepth2]) pDepth2 = renderData[kDepth2]->asTexture();
    else if (mDepthMode == DepthMode::DualDepth) mDepthMode = DepthMode::SingleDepth;
    Texture::SharedPtr psDepth;
    if (renderData[ksDepth]) psDepth = renderData[ksDepth]->asTexture();
    else if (mDepthMode == DepthMode::StochasticDepth) mDepthMode = DepthMode::SingleDepth;
    auto pMaterial = renderData[kMaterial]->asTexture();

    auto pInternalRasterDepth = renderData[kInternalRasterDepth]->asTexture();
    auto pInternalRayDepth = renderData[kInternalRayDepth]->asTexture();
    //auto pInternalInstanceID = renderData[kInternalInstanceID]->asTexture();
    auto pInternalAskRay = renderData[kInternalAskRay]->asTexture();
    auto pInternalRequireRay = renderData[kInternalRequireRay]->asTexture();
    auto pInternalForceRay = renderData[kInternalForceRay]->asTexture();
    auto pInternalRasterAO = renderData[kInternalRasterAO]->asTexture();
    auto pInternalRayAO = renderData[kInternalRayAO]->asTexture();
    auto pInternalSphereEnd = renderData[kInternalSphereEnd]->asTexture();

    auto pCamera = mpScene->getCamera().get();

    if (mEnabled)
    {
        if (mClearTexture)
        {
            pRenderContext->clearTexture(pAoDst.get(), float4(0.0f));
            mClearTexture = false;
        }

        if (!mpSSAOPass)
        {
            Program::Desc desc;
            desc.addShaderModules(mpScene->getShaderModules());
            desc.addShaderLibrary(kSSAOShader).psEntry("main");
            desc.addTypeConformances(mpScene->getTypeConformances());
            desc.setShaderModel("6_5");
            // program defines
            Program::DefineList defines;
            defines.add(mpScene->getSceneDefines());
            defines.add("DEPTH_MODE", std::to_string(uint32_t(mDepthMode)));
            defines.add("KERNEL_SIZE", std::to_string(mKernelSize));
            defines.add("PREVENT_DARK_HALOS", mPreventDarkHalos ? "1" : "0");
            if (psDepth) defines.add("MSAA_SAMPLES", std::to_string(psDepth->getSampleCount()));

            mpSSAOPass = FullScreenPass::create(mpDevice, desc, defines);
            mDirty = true;

            mpSSAOPass["gRasterDepth"] = pInternalRasterDepth;
            mpSSAOPass["gRayDepth"] = pInternalRayDepth;
            mpSSAOPass["gRasterAO"] = pInternalRasterAO;
            mpSSAOPass["gRayAO"] = pInternalRayAO;
            mpSSAOPass["gSphereEnd"] = pInternalSphereEnd;
            mpSSAOPass["gAskRay"] = pInternalAskRay;
            mpSSAOPass["gRequireRay"] = pInternalRequireRay;
            mpSSAOPass["gForceRay"] = pInternalForceRay;
            //mpSSAOPass["gInstanceIDOut"] = pInternalInstanceID;
        }

        if (mDirty)
        {
            // bind static resources
            mData.noiseScale = float2(pDepth->getWidth(), pDepth->getHeight()) / float2(NOISE_SIZE, NOISE_SIZE);
            mpSSAOPass["StaticCB"].setBlob(mData);
            mDirty = false;
        }

        // bind dynamic resources
        auto var = mpSSAOPass->getRootVar();
        mpScene->setRaytracingShaderData(pRenderContext, var);

        pCamera->setShaderData(mpSSAOPass["PerFrameCB"]["gCamera"]);
        mpSSAOPass["PerFrameCB"]["saveDepths"] = mSaveDepths;
        mpSSAOPass["PerFrameCB"]["invViewMat"] = inverse(pCamera->getViewMatrix());

        // Update state/vars
        mpSSAOPass["gNoiseSampler"] = mpNoiseSampler;
        mpSSAOPass["gTextureSampler"] = mpTextureSampler;
        mpSSAOPass["gDepthTex"] = pDepth;
        mpSSAOPass["gDepthTex2"] = pDepth2;
        mpSSAOPass["gsDepthTex"] = psDepth;
        mpSSAOPass["gNoiseTex"] = mpNoiseTexture;
        mpSSAOPass["gNormalTex"] = pNormals;
        //mpSSAOPass["gInstanceID"] = pInstanceID;
        if (mSaveDepths)
        {
            mpSSAOPass["gMaterialData"] = pMaterial;

            // clear uav targets
            pRenderContext->clearTexture(pInternalRasterDepth.get());
            pRenderContext->clearTexture(pInternalRayDepth->asTexture().get());
            //pRenderContext->clearTexture(pInternalInstanceID->asTexture().get());
            //pRenderContext->clearUAV(pInternalInstanceID->asTexture()->getUAV().get(), uint4(0));
            pRenderContext->clearUAV(pInternalRequireRay->asTexture()->getUAV().get(), uint4(0));
            pRenderContext->clearUAV(pInternalAskRay->asTexture()->getUAV().get(), uint4(0));
            pRenderContext->clearUAV(pInternalForceRay->asTexture()->getUAV().get(), uint4(0));
            pRenderContext->clearTexture(pInternalRasterAO->asTexture().get());
            pRenderContext->clearTexture(pInternalRayAO->asTexture().get());
            pRenderContext->clearTexture(pInternalSphereEnd->asTexture().get());
        }

        // Generate AO
        mpAOFbo->attachColorTarget(pAoDst, 0);
        setGuardBandScissors(*mpSSAOPass->getState(), renderData.getDefaultTextureDims(), mGuardBand);
        mpSSAOPass->execute(pRenderContext, mpAOFbo, false);

        if (mSaveDepths)
        {
            // write sample information
            pInternalRasterDepth->captureToFile(0, -1, "ML/raster.dds", Bitmap::FileFormat::DdsFile);
            pInternalRayDepth->captureToFile(0, -1, "ML/ray.dds", Bitmap::FileFormat::DdsFile);
            //pInternalInstanceID->captureToFile(0, -1, "instance.dds", Bitmap::FileFormat::DdsFile);
            //pInstanceID->captureToFile(0, -1, "instance_center.dds", Bitmap::FileFormat::DdsFile);
            pInternalForceRay->captureToFile(0, -1, "ML/forceRay.dds", Bitmap::FileFormat::DdsFile);
            pInternalRequireRay->captureToFile(0, -1, "ML/requireRay.dds", Bitmap::FileFormat::DdsFile);
            pInternalAskRay->captureToFile(0, -1, "ML/askRay.dds", Bitmap::FileFormat::DdsFile);
            pInternalRasterAO->captureToFile(0, -1, "ML/rasterAO.dds", Bitmap::FileFormat::DdsFile);
            pInternalRayAO->captureToFile(0, -1, "ML/rayAO.dds", Bitmap::FileFormat::DdsFile);
            pInternalSphereEnd->captureToFile(0, -1, "ML/sphereEnd.dds", Bitmap::FileFormat::DdsFile);

            auto sphereHeights = getSphereHeights();

            // TODO vao_to_numpy(sphereHeights, "ML/raster.dds", "ML/ray.dds", "ML/askRay.dds", "ML/requireRay.dds", "ML/forceRay.dds", "ML/rasterAO.dds", "ML/rayAO.dds", "ML/sphereEnd.dds", mTrainingIndex, mIsTraining);
            mTrainingIndex++;

            mSaveDepths = false;
        }
    }
    else // ! enabled
    {
        pRenderContext->clearTexture(pAoDst.get(), float4(1.0f));
    }
}

void VAO::renderUI(Gui::Widgets& widget)
{
    widget.checkbox("Enabled", mEnabled);
    if (!mEnabled) return;

    if (widget.checkbox("Prevent Dark Halos", mPreventDarkHalos))
    {
        mpSSAOPass.reset();
    }

    if (widget.var("Guard Band", mGuardBand, 0, 256)) mClearTexture = true;

    uint32_t depthMode = (uint32_t)mDepthMode;
    if (widget.dropdown("Depth Mode", kDepthModeDropdown, depthMode)) {
        mDepthMode = (DepthMode)depthMode;
        mpSSAOPass.reset();
    }

    uint32_t distribution = (uint32_t)mHemisphereDistribution;
    if (widget.dropdown("Kernel Distribution", kDistributionDropdown, distribution))
    {
        mHemisphereDistribution = (SampleDistribution)distribution;
        setKernel();
    }

    uint32_t kernelSize = mKernelSize;
    if (widget.var("Kernel Size", kernelSize, 1u, VAOData::kMaxSamples))
    {
        kernelSize = glm::clamp(kernelSize, 1u, VAOData::kMaxSamples);
        mKernelSize = kernelSize;
        setKernel();
        requestRecompile();
    }
    
    if (widget.var("Sample Radius", mData.radius, 0.01f, FLT_MAX, 0.01f)) mDirty = true;

    if (widget.var("Thickness", mData.thickness, 0.0f, 1.0f, 0.1f)) mDirty = true;

    if (widget.var("Power Exponent", mData.exponent, 1.0f, 4.0f, 0.1f)) mDirty = true;

    if (widget.checkbox("Training Data", mIsTraining))
    {
        mTrainingIndex = 0;
    }

    widget.text("Training Index: " + std::to_string(mTrainingIndex));

    if (widget.button("Save Depths")) mSaveDepths = true;
}

void VAO::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    mpScene = pScene;
    mDirty = true;
    mpSSAOPass.reset();
}


void VAO::setKernel()
{
    std::srand(5960372); // same seed for kernel
    int vanDerCorputOffset = mKernelSize; // (only correct for power of two numbers => offset 8 results in 1/16, 9/16, 5/16... which are 8 different uniformly dstributed numbers, see https://en.wikipedia.org/wiki/Van_der_Corput_sequence)
    bool isPowerOfTwo = std::_Popcount(uint32_t(vanDerCorputOffset)) == 1;//std::has_single_bit(uint32_t(vanDerCorputOffset));
    if (mHemisphereDistribution == SampleDistribution::VanDerCorput && !isPowerOfTwo)
        logWarning("VanDerCorput sequence only works properly if the sample count is a power of two!");

    if (mHemisphereDistribution == SampleDistribution::Poisson)
    {
        // brute force algorithm to generate poisson samples
        float r = 0.28f; // for kernelSize = 8
        if (mKernelSize >= 16) r = 0.19f;
        if (mKernelSize >= 24) r = 0.15f;
        if (mKernelSize >= 32) r = 0.13f;

        auto pow2 = [](float x) {return x * x; };

        uint i = 0; // current length of list
        uint cur_attempt = 0;
        while (i < mKernelSize)
        {
            i = 0; // reset list length
            const uint max_retries = 10000;
            uint cur_retries = 0;
            while (i < mKernelSize && cur_retries < max_retries)
            {
                cur_retries += 1;
                float2 point = float2(glm::linearRand(-1.0f, 1.0f), glm::linearRand(-1.0f, 1.0f));
                if (point.x * point.x + point.y * point.y > pow2(1.0f - r))
                    continue;

                bool too_close = false;
                for (uint j = 0; j < i; ++j)
                    if (pow2(point.x - mData.sampleKernel[j].x) + pow2(point.y - mData.sampleKernel[j].y) < pow2(2.0f * r))
                    {
                        too_close = true;
                        break;
                    }


                if (too_close) continue;

                mData.sampleKernel[i++] = float4(point.x, point.y, glm::linearRand(0.0f, 1.0f), glm::linearRand(0.0f, 1.0f));
            }

            std::cerr << "\rpoisson attempt " << ++cur_attempt;

            if (cur_attempt % 1000 == 0)
                r = r - 0.01f; // shrink radius every 1000 attempts
        }

        // succesfully found points

    }
    else if (mHemisphereDistribution == SampleDistribution::Triangle)
    {
        float2 p0 = float2(0.0f, 0.8f);
        float2 p1 = float2(0.6f, -0.6f);
        float2 p2 = float2(-0.6f, -0.6f);

        for (uint32_t i = 0; i < mKernelSize; i++)
        {
            float2 rand;
            // select two points
            if ((float)i / (float)mKernelSize * 8.0f < 3.0f)
            {
                rand = glm::mix(p0, p1, (float)i / (float)mKernelSize * 8.0f / 3.0f);
            }
            else if ((float)i / (float)mKernelSize * 8.0f < 5.0f)
            {
                rand = glm::mix(p1, p2, ((float)i / (float)mKernelSize * 8.0f - 3.0f) / 2.0f);
            }
            else
            {
                rand = glm::mix(p2, p0, ((float)i / (float)mKernelSize * 8.0f - 5.0f) / 3.0f);
            }
            mData.sampleKernel[i].x = rand.x;
            mData.sampleKernel[i].y = rand.y;
            mData.sampleKernel[i].z = glm::linearRand(0.0f, 1.0f);
            mData.sampleKernel[i].w = glm::linearRand(0.0f, 1.0f);
        }
    }
    else // random or hammersly
    {
        std::string nums;
        for (uint32_t i = 0; i < mKernelSize; i++)
        {
            auto& s = mData.sampleKernel[i];
            float2 rand;
            switch (mHemisphereDistribution)
            {
            case SampleDistribution::Random:
                rand = float2(glm::linearRand(0.0f, 1.0f), glm::linearRand(0.0f, 1.0f));
                break;

            case SampleDistribution::VanDerCorput:
                // skip 0 because it will results in (0, 0) which results in sample point (0, 0)
                // => this means that we sample the same position for all tangent space rotations
                rand = float2((float)(i) / (float)(mKernelSize), radicalInverse(vanDerCorputOffset + i));
                break;

            default: throw std::runtime_error("unknown kernel distribution");
            }

            const float max_radius = 1.0f;

            float theta = rand.x * 2.0f * glm::pi<float>();
            float r = glm::sqrt(1.0f - glm::pow(rand.y, 2.0f / 3.0f)) * max_radius;
            nums += std::to_string(r) + ", ";
            s.x = r * sin(theta);
            s.y = r * cos(theta);
            s.z = glm::linearRand(0.0f, 1.0f);
            s.w = glm::linearRand(0.0f, 1.0f);
        }
    }


    mDirty = true;
}

std::vector<float> VAO::getSphereHeights() const
{
    std::vector<float> heights;
    heights.reserve(mKernelSize);
    for (uint32_t i = 0; i < mKernelSize; i++)
    {
        auto rand = float2(mData.sampleKernel[i].x, mData.sampleKernel[i].y);
        float sphereHeight = glm::sqrt(1.0f - dot(rand, rand));
        heights.push_back(sphereHeight);
    }
    return heights;
}

void VAO::setNoiseTexture()
{
    mDirty = true;

    std::vector<uint8_t> data;
    data.resize(NOISE_SIZE * NOISE_SIZE);

    // https://en.wikipedia.org/wiki/Ordered_dithering
    const float ditherValues[] = { 0.0f, 8.0f, 2.0f, 10.0f, 12.0f, 4.0f, 14.0f, 6.0f, 3.0f, 11.0f, 1.0f, 9.0f, 15.0f, 7.0f, 13.0f, 5.0f };

    for (uint32_t i = 0; i < data.size(); i++)
    {
        data[i] = uint8_t(ditherValues[i] / 16.0f * 255.0f);
    }

    mpNoiseTexture = Texture::create2D(mpDevice.get(), NOISE_SIZE, NOISE_SIZE, ResourceFormat::R8Unorm, 1, 1, data.data());
}
