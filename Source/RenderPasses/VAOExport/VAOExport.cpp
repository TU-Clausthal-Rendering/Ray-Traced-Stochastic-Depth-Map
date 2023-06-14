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
#include "VAOExport.h"

namespace
{
    const std::string kRef = "ref";
    const std::string kBright = "bright";
    const std::string kDark = "dark";
    const std::string kDepth = "depth";
    const std::string kDepthInv = "invDepth"; // inverse depth (1/z)
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, VAOExport>();
}

VAOExport::VAOExport(ref<Device> pDevice, const Dictionary& dict)
    : RenderPass(pDevice)
{
}

Dictionary VAOExport::getScriptingDictionary()
{
    return Dictionary();
}

RenderPassReflection VAOExport::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addInput(kRef, "reference").texture2D(0, 0, 1, 1, mArraySize);
    reflector.addInput(kBright, "bright").texture2D(0, 0, 1, 1, mArraySize);
    reflector.addInput(kDark, "dark").texture2D(0, 0, 1, 1, mArraySize);
    reflector.addInput(kDepth, "depth").texture2D(0, 0, 1, 1, mArraySize);
    reflector.addInput(kDepthInv, "non-linear z").texture2D(0, 0, 1, 1, mArraySize);
    reflector.addOutput("dummy", "dummy");
    return reflector;
}

void VAOExport::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto pRefTex = renderData[kRef]->asTexture();
    auto pBrightTex = renderData[kBright]->asTexture();
    auto pDarkTex = renderData[kDark]->asTexture();
    auto pDepthTex = renderData[kDepth]->asTexture();
    auto pDepthInvTex = renderData[kDepthInv]->asTexture();

    if(mSave)
    {
        for(uint slice = 0; slice < mArraySize; ++slice)
        {
            pRefTex->captureToFile(0, slice, getExportName("ref", ".png", slice), Bitmap::FileFormat::PngFile);
            pBrightTex->captureToFile(0, slice, getExportName("bright", ".png", slice), Bitmap::FileFormat::PngFile);
            pDarkTex->captureToFile(0, slice, getExportName("dark", ".png", slice), Bitmap::FileFormat::PngFile);
            pDepthTex->captureToFile(0, slice, getExportName("depth", ".exr", slice), Bitmap::FileFormat::ExrFile);
            pDepthInvTex->captureToFile(0, slice, getExportName("invDepth", ".exr", slice), Bitmap::FileFormat::ExrFile);
        }

        mExportIndex++;
        mSave = false;
    }
}

void VAOExport::renderUI(Gui::Widgets& widget)
{
    widget.textbox("Export Directory", mExportFolder);
    widget.var("Number", mExportIndex);
    if (widget.button("Save")) mSave = true;
}

std::string VAOExport::getExportName(const std::string& type, const std::string& extension, uint slice)
{
    return mExportFolder + type + "_" + std::to_string(mExportIndex) + "_s" + std::to_string(slice) + extension;
}
