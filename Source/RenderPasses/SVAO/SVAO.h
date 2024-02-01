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
#pragma once
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "../VAO/DepthMode.h"
#include "VAOData.slang"
#include "NeuralNet.h"
#include "Core/Pass/FullScreenPass.h"
#include "../StochasticDepthMap/StochasticDepthImplementation.h"
#include "AOKernel.h"

using namespace Falcor;

class SVAO : public RenderPass
{
public:
    enum StochasticDepthImpl
    {
        Raster = 0,
        Ray = 1
    };

    FALCOR_PLUGIN_CLASS(SVAO, "SVAO", "Stenciled Volumetric Ambient Occlusion");

    /** Create a new render pass object.
        \param[in] pDevice GPU device.
        \param[in] dict Properties of serialized parameters.
        \return A new object, or an exception is thrown if creation failed.
    */
    static ref<SVAO> create(ref<Device> pDevice, const Properties& dict);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }
    SVAO(ref<Device> pDevice);
private:
    ref<Texture> genNoiseTexture();

    Program::Desc getFullscreenShaderDesc(const std::string& filename);

    ref<Sampler> mpNoiseSampler;
    ref<Texture> mpNoiseTexture;

    ref<Sampler> mpTextureSampler;
    ref<ComputePass> mpComputePass;

    ref<Scene> mpScene;

    ref<RenderGraph> mpStochasticDepthGraph;

    // 2nd pass
    ref<ComputePass> mpComputePass2;

    ref<RtProgram> mpRayProgram;
    ref<RtProgramVars> mRayVars;

    uint mStochSamples = 4; // for stochastic depth map


    // general settings
    bool mEnabled = true;

    VAOData mData;
    bool mDirty = true;
    uint2 mStochLastSize;
    DepthMode mPrimaryDepthMode = DepthMode::SingleDepth;
    DepthMode mSecondaryDepthMode = DepthMode::Raytraced;
    bool mUseRayPipeline = true;

    // performance knobs
    bool mTraceOutOfScreen = true;

    int mFrameIndex = 0;

    //NeuralNetCollection mNeuralNet;
    //NeuralNetCollection mNeuralNet2 = NeuralNetCollection(NeuralNetCollection::Type::Regressor);

    StochasticDepthImpl mStochasticDepthImpl = StochasticDepthImpl::Ray;
    
    uint mStochMapDivisor = 1; // 1 = same resolution, 2 = half resolution etc.
    bool mStochMapNormals = false;
    bool mStochMapJitter = true; // very cheap

    bool mDualAo = false; // if true, AO will be two channel, one for bright and one for dark
    bool mAlphaTest = true; // use alpha test in raytracing or stochastic depth passes
    StochasticDepthImplementation mStochasticDepthImplementation = StochasticDepthImplementation::Default;
    bool mUseRayInterval = true; // stochastic depth ray interval optimization
    AOKernel mKernel = AOKernel::VAO;
    RasterizerState::CullMode mCullMode = RasterizerState::CullMode::None; // cull mode for secondary surfaces
};
