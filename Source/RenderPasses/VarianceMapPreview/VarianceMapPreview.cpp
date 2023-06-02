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
#include "VarianceMapPreview.h"

namespace 
{
    const std::string kDepth = "depthvm";
    const std::string kNormal = "normalvm";
    const std::string kOutput = "output";

    const std::string kShaderFile = "RenderPasses/VarianceMapPreview/VarianceMapPreview.ps.slang";

    const std::string kMode = "mode";
}

static void regModules(pybind11::module& m)
{
    // register PreviewMode enum
    pybind11::enum_<VarianceMapPreview::PreviewMode> previewMode(m, "VariancePreviewMode");
    previewMode.value("Depth", VarianceMapPreview::PreviewMode::Depth);
    previewMode.value("Normal", VarianceMapPreview::PreviewMode::Normal);
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, VarianceMapPreview>();
    ScriptBindings::registerBinding(regModules);
}

VarianceMapPreview::VarianceMapPreview(ref<Device> pDevice, const Dictionary& dict)
    : RenderPass(pDevice)
{
    mpFbo = Fbo::create(pDevice);
    mpPass = FullScreenPass::create(mpDevice, kShaderFile);

    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
    mpPass->getRootVar()["S"] = Sampler::create(pDevice, samplerDesc);
}

Dictionary VarianceMapPreview::getScriptingDictionary()
{
    return Dictionary();
}

RenderPassReflection VarianceMapPreview::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addInput(kDepth, "depth variance map")
        .texture2D(0, 0, 1, 0);
    reflector.addInput(kNormal, "normal variance map")
        .texture2D(0, 0, 1, 0);
    reflector.addOutput(kOutput, "output image").format(ResourceFormat::RGBA32Float).bindFlags(Resource::BindFlags::AllColorViews);
    return reflector;
}

void VarianceMapPreview::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto pDepth = renderData[kDepth]->asTexture();
    auto pNormal = renderData[kNormal]->asTexture();
    auto pDst = renderData[kOutput]->asTexture();

    auto vars = mpPass->getRootVar();
    vars["CBuffer"]["mode"] = (uint32_t)mMode;
    vars["CBuffer"]["level"] = mSelectedMip;
    if (mMode == PreviewMode::Depth)
    {
        vars["gInput"] = pDepth;
    }
    else
    {
        vars["gInput"] = pNormal;
    }

    mpFbo->attachColorTarget(pDst, 0);
    mpPass->execute(pRenderContext, mpFbo);
}

void VarianceMapPreview::renderUI(Gui::Widgets& widget)
{
    const Gui::DropdownList kPreviewModeList =
    {
        { (uint32_t)PreviewMode::Depth, "Depth" },
        { (uint32_t)PreviewMode::Normal, "Normal" },
    };
    
    widget.dropdown("Preview Mode", kPreviewModeList, reinterpret_cast<uint32_t&>(mMode));
    widget.var("Mip Level", mSelectedMip, 0, 15);
}
