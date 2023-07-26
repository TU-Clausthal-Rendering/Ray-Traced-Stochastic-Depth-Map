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

using namespace Falcor;

class TemporalDepthPeel : public RenderPass
{
public:
    enum class Implementation
    {
        Iterative,
        Raster
    };

    FALCOR_ENUM_INFO(Implementation,
    {
           { Implementation::Iterative, "Iterative" },
           { Implementation::Raster, "Raster" }
    });

    FALCOR_PLUGIN_CLASS(TemporalDepthPeel, "TemporalDepthPeel", "Insert pass description here.");

    static ref<TemporalDepthPeel> create(ref<Device> pDevice, const Properties& props) { return make_ref<TemporalDepthPeel>(pDevice, props); }

    TemporalDepthPeel(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

private:
    ref<Texture> allocatePrevFrameTexture(const ref<Texture>& original, ref<Texture> prev) const;
    ref<Buffer> genIndexBuffer(uint2 res) const;

    ref<FullScreenPass> mpIterPass;
    ref<Fbo> mpFbo;

    ref<Texture> mpPrevDepth;
    ref<Texture> mpPrevDepth2;

    ref<Scene> mpScene;
    bool mEnabled = true;
    Implementation mImplementation = Implementation::Raster;

    ref<Buffer> mRasterIndexBuffer;
    ref<GraphicsState> mpRasterState;
    ref<GraphicsVars> mpRasterVars;
};

FALCOR_ENUM_REGISTER(TemporalDepthPeel::Implementation);
