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
#include "GuardBand.h"

#include "RenderGraph/RenderPassStandardFlags.h"

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, GuardBand>();
}

GuardBand::GuardBand(ref<Device> pDevice, const Properties& dict)
    : RenderPass(pDevice)
{
    mGuardBand = dict.get("guardBand", mGuardBand);
}

Properties GuardBand::getProperties() const
{
    Properties d;
    d["guardBand"] = mGuardBand;
    return d;
}

RenderPassReflection GuardBand::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    return reflector;
}

void GuardBand::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto& dict = renderData.getDictionary();
    dict["guardBand"] = mGuardBand;
    dict["guardBand.uvMin"] = float2(float(mGuardBand) + 0.5f) / float2(renderData.getDefaultTextureDims());
    dict["guardBand.uvMax"] = (float2(renderData.getDefaultTextureDims()) - float2(float(mGuardBand) + 0.5f)) / float2(renderData.getDefaultTextureDims());

    mApp = dict[kRenderer];
}

void GuardBand::renderUI(Gui::Widgets& widget)
{
    widget.var("Guard  Band", mGuardBand, 0, 256);
    if (widget.button("Recompile"))
    {
        requestRecompile();
    }
    if(widget.button("Resize 1920x1080") && mApp)
    {
        mApp->resizeFrameBuffer(1920 + 2 * mGuardBand, 1080 + 2 * mGuardBand);
    }
    if (widget.button("Resize 800x600") && mApp)
    {
        mApp->resizeFrameBuffer(800 + 2 * mGuardBand, 600 + 2 * mGuardBand);
    }
}
