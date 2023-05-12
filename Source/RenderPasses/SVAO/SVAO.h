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
#include "../VAO/DepthMode.h"
#include "VAOData.slang"
#include "NeuralNet.h"
#include "RenderGraph/BasePasses/FullScreenPass.h"

using namespace Falcor;

class SVAO : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(SVAO, "SVAO", "Stenciled Volumetric Ambient Occlusion");

    using SharedPtr = std::shared_ptr<SVAO>;

    /** Create a new render pass object.
        \param[in] pDevice GPU device.
        \param[in] dict Dictionary of serialized parameters.
        \return A new object, or an exception is thrown if creation failed.
    */
    static SharedPtr create(std::shared_ptr<Device> pDevice, const Dictionary& dict);

    virtual Dictionary getScriptingDictionary() override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

private:
    SVAO(std::shared_ptr<Device> pDevice);
    Texture::SharedPtr genNoiseTexture();

    Program::Desc getFullscreenShaderDesc(const std::string& filename);

    Fbo::SharedPtr mpFbo;

    Sampler::SharedPtr mpNoiseSampler;
    Texture::SharedPtr mpNoiseTexture;

    Sampler::SharedPtr mpTextureSampler;
    FullScreenPass::SharedPtr mpRasterPass;

    Scene::SharedPtr mpScene;
    bool mEnableRayFilter = false;

    std::shared_ptr<RenderGraph> mpStochasticDepthGraph;
    //RayFilter::SharedPtr mpRayFilter;

    // 2nd pass
    Fbo::SharedPtr mpFbo2;
    FullScreenPass::SharedPtr mpRasterPass2;
    DepthStencilState::SharedPtr mpDepthStencilState;

    FullScreenPass::SharedPtr mpStencilPass;
    Fbo::SharedPtr mpStencilFbo;

    RtProgram::SharedPtr mpRayProgram;
    RtProgramVars::SharedPtr mRayVars;

    int msaa_sample = 4; // for stochastic depth map


    // general settings
    bool mEnabled = true;

    VAOData mData;
    bool mDirty = true;
    uint2 lastSize;
    int mGuardBand = 64;
    DepthMode mPrimaryDepthMode = DepthMode::SingleDepth;
    DepthMode mSecondaryDepthMode = DepthMode::Raytraced;
    bool mUseRayPipeline = true;

    bool mPreventDarkHalos = true;

    // performance knobs
    bool mTraceOutOfScreen = true;
    bool mTraceDoubleOnDouble = false;
    float mClassifyProbability = 0.5;

    NeuralNetCollection mNeuralNet;
    NeuralNetCollection mNeuralNet2 = NeuralNetCollection(NeuralNetCollection::Type::Regressor);
};
