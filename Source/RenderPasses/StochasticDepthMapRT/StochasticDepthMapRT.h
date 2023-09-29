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
#include "Core/Pass/FullScreenPass.h"
#include "RenderGraph/RenderPass.h"
#include "../StochasticDepthMap/StochasticDepthImplementation.h"

using namespace Falcor;

class StochasticDepthMapRT : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(StochasticDepthMapRT, "StochasticDepthMapRT", "Captures mutliple depth layers stochastically inside a MSAA texture with raytracing");

    /** Create a new render pass object.
        \param[in] pDevice GPU device.
        \param[in] dict Properties of serialized parameters.
        \return A new object, or an exception is thrown if creation failed.
    */
    static ref<StochasticDepthMapRT> create(ref<Device> pDevice, const Properties& dict);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }
    StochasticDepthMapRT(ref<Device> pDevice);
private:

    ref<RtProgram> mpRayProgram;
    ref<RtProgramVars> mRayVars;

    ref<FullScreenPass> mpRasterProgram;
    ref<Fbo> mpFbo;

    ref<FullScreenPass> mpStencilPass;
    ref<DepthStencilState> mpStencilState;

    RasterizerState::CullMode mCullMode = RasterizerState::CullMode::Back;
    uint32_t mSampleCount = 4;
    bool mNormalize = true; // normalize to range [0, 1] (based on [near, far])

    ref<Scene> mpScene;

    ref<Buffer> mpStratifiedLookUpBuffer;
    ref<Buffer> mpStratifiedIndices;
    ref<Buffer> mpMaterialAlphaTest;

    bool mClear = false;
    bool mUseRayPipeline = true;
    bool mAlphaTest = true;

    bool mUse16Bit = false; // compresses depth to 16 bit, otherwise 32 bit
    bool mStoreNormals = false; // adds normals to each sample point
    bool mJitter = false; // jitter the pixel positions, can help with aliasing
    StochasticDepthImplementation mImplementation = StochasticDepthImplementation::Default;
    float mAlpha = 0.2f; // for coverage mask implementation
};
